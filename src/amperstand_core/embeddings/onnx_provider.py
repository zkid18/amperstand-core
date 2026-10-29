"""In-process ONNX embeddings — the "Nearby (c)" mode.

Optional extra: `pip install 'amperstand-core[onnx]'` pulls onnxruntime,
tokenizers and numpy. Nothing here is imported until the provider is
actually constructed, so the base install stays free of onnxruntime.

Expects a Hugging Face-style model directory with `tokenizer.json` and
an ONNX graph whose inputs are `input_ids`/`attention_mask` (optionally
`token_type_ids`) and whose first output is either `last_hidden_state`
(pooled here: CLS or mean) or an already-pooled `sentence_embedding`.
Tested with ibm-granite/granite-embedding-97m-multilingual-r2
(`onnx/model_quint8_avx2.onnx`, CLS pooling, 384-d).

Threads default to 1 — on the 1 vCPU droplet more threads only add
contention with uvicorn.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from amperstand_core.embeddings.base import BaseEmbedder, EmbeddingConfigError, EmbeddingError

logger = logging.getLogger(__name__)

DEFAULT_ONNX_FILE = "onnx/model.onnx"
INSTALL_HINT = "install the optional extra: pip install 'amperstand-core[onnx]'"


class ONNXEmbedder(BaseEmbedder):
    provider = "onnx"

    def __init__(
        self,
        *,
        model_dir: str | os.PathLike,
        onnx_file: str = DEFAULT_ONNX_FILE,
        dim: int,
        model: str | None = None,
        pooling: str = "cls",
        threads: int = 1,
        max_model_tokens: int = 2048,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingConfigError(f"onnx provider needs onnxruntime+tokenizers+numpy; {INSTALL_HINT}") from exc

        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise EmbeddingConfigError(f"onnx model dir not found: {self.model_dir}")
        if pooling not in ("cls", "mean"):
            raise EmbeddingConfigError(f"onnx pooling must be cls|mean, got {pooling!r}")
        super().__init__(model=model or self.model_dir.name, dim=dim, **kwargs)
        self.pooling = pooling
        self.normalize = normalize
        self.max_model_tokens = max_model_tokens

        tok_path = self.model_dir / "tokenizer.json"
        graph_path = self.model_dir / onnx_file
        if not tok_path.exists():
            raise EmbeddingConfigError(f"missing {tok_path}")
        if not graph_path.exists():
            raise EmbeddingConfigError(f"missing {graph_path}")

        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.no_padding()
        self._tok.enable_truncation(max_model_tokens)
        self._pad_id = self._pad_token_id()

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(graph_path), so, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._output_name = self._sess.get_outputs()[0].name
        logger.info("onnx embedder loaded %s (%s, pooling=%s, threads=%d)",
                    self.model, onnx_file, pooling, threads)

    def _pad_token_id(self) -> int:
        for cand in ("<pad>", "[PAD]", "<|endoftext|>", "</s>"):
            tid = self._tok.token_to_id(cand)
            if tid is not None:
                return tid
        return 0

    def describe(self) -> dict[str, object]:
        d = super().describe()
        d.update({"model_dir": str(self.model_dir), "pooling": self.pooling})
        return d

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        import numpy as np

        try:
            encs = self._tok.encode_batch(texts)
            L = max(len(e.ids) for e in encs)
            ids = np.full((len(encs), L), self._pad_id, dtype=np.int64)
            mask = np.zeros((len(encs), L), dtype=np.int64)
            for r, e in enumerate(encs):
                ids[r, : len(e.ids)] = e.ids
                mask[r, : len(e.ids)] = 1
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            out = self._sess.run([self._output_name], feeds)[0]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"onnx inference failed: {exc}") from exc

        if out.ndim == 3:  # last_hidden_state → pool
            if self.pooling == "cls":
                emb = out[:, 0, :]
            else:
                m = mask[..., None].astype(np.float32)
                emb = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-6, None)
        else:
            emb = out
        emb = emb.astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-12, None)
        return emb.tolist()
