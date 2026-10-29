"""Re-ranking for the final stage of hybrid search.

After BM25 + vector + RRF give us ~30-60 candidate sections, a reranker
sees the query and each passage together and reorders them. Three kinds:

  llm            the original grader: a chat model (gpt-4o-mini by default,
                 or any OpenAI-compatible chat endpoint) rates every passage
                 0-10 in one JSON-mode call. ~$0.001/query.
  cross_encoder  a scoring model behind an HTTP `/rerank` API — a TEI or
                 Infinity sidecar serving mmarco-mMiniLMv2 ("Nearby"), or a
                 vendor API: Cohere, Jina, Voyage ("BYOK") — or an in-process
                 ONNX cross-encoder (optional extra).
  none           rerank is not offered; requests that ask for it get a 503.

Failure policy is explicit: `soft` (default) returns the fused order and
flags `rerank_applied=false` in the response meta; `hard` raises so the
API answers 503. Neither is silent.

Configuration (env):
  AMPERSTAND_RERANK_KIND        llm (default) | cross_encoder | none
  AMPERSTAND_RERANK_ENABLED     kill switch (0 → behaves like none)
  AMPERSTAND_RERANK_PROVIDER    llm: openai (default) | openai_compatible
                                cross_encoder: tei | infinity | cohere | jina | voyage | onnx
  AMPERSTAND_RERANK_MODEL       chat model (llm, default gpt-4o-mini) or rerank model
  AMPERSTAND_RERANK_BASE_URL    endpoint host (required for openai_compatible / tei / infinity)
  AMPERSTAND_RERANK_API_KEY     key; falls back to OPENAI_API_KEY / COHERE_API_KEY /
                                JINA_API_KEY / VOYAGE_API_KEY
  AMPERSTAND_RERANK_MAX_CHARS   passage cap sent to the reranker (default 600)
  AMPERSTAND_RERANK_TIMEOUT_S   HTTP timeout (default 20)
  AMPERSTAND_RERANK_FAIL_MODE   soft (default) | hard
  AMPERSTAND_RERANK_ONNX_DIR / _ONNX_FILE / _ONNX_THREADS / _ONNX_MAX_TOKENS   (provider=onnx)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import httpx

from amperstand_core.search.models import SearchResult

logger = logging.getLogger(__name__)

# gpt-4o-mini is the sweet spot for re-ranking: cheap enough at the
# personal-vault query rate, smart enough to grade short passages on a
# numeric scale, and stable across runs at temperature=0.
RERANK_MODEL = os.environ.get("AMPERSTAND_RERANK_MODEL", "gpt-4o-mini")
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Per-candidate body cap. ~600 chars ≈ 150 tokens — long enough to be
# graded fairly, short enough that 30 candidates fit in one request.
RERANK_MAX_CHARS = 600

KINDS = ("llm", "cross_encoder", "none")
CROSS_ENCODER_PROVIDERS = ("tei", "infinity", "cohere", "jina", "voyage", "onnx")
LLM_PROVIDERS = ("openai", "openai_compatible")

COHERE_RERANK_URL = "https://api.cohere.com/v2"
JINA_RERANK_URL = "https://api.jina.ai/v1"
VOYAGE_RERANK_URL = "https://api.voyageai.com/v1"

BodyLookup = Callable[[SearchResult], str]


class RerankError(RuntimeError):
    """The reranker could not produce an order (provider down, bad key, ...)."""


def rerank_enabled() -> bool:
    """Server-wide off switch. Operations can disable rerank via env var."""
    return os.environ.get("AMPERSTAND_RERANK_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


@dataclass
class RerankOutcome:
    results: list[SearchResult]
    applied: bool
    kind: str
    error: str | None = None


# ── shared helpers ───────────────────────────────────────────────────


def _passages(
    candidates: Sequence[SearchResult], body_lookup: BodyLookup, max_chars: int,
) -> list[dict[str, Any]]:
    items = []
    for i, c in enumerate(candidates):
        try:
            body = body_lookup(c)
        except Exception:  # noqa: BLE001
            body = ""
        if not body:
            body = c.snippet or ""
        items.append({
            "id": i,
            "title": c.section_title or "",
            "path": " > ".join(c.section_path) if c.section_path else "",
            "text": body[:max_chars],
        })
    return items


def _passage_text(item: dict[str, Any]) -> str:
    """Flatten a passage for cross-encoders: heading context + body."""
    head = item["path"] or item["title"]
    return f"{head}\n{item['text']}" if head else item["text"]


def _apply_scores(
    candidates: Sequence[SearchResult], scores: Mapping[int, float], limit: int,
) -> list[SearchResult]:
    """Re-order by score desc; unscored candidates keep their fused position."""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda p: (-scores.get(p[0], float("-inf")), p[0]))
    return [
        # Surface the rerank score on the SearchResult so the UI can show
        # "why this came back first" — negative so lower=better matches
        # the rest of the codebase's score convention.
        SearchResult(
            doc_id=c.doc_id,
            section_title=c.section_title,
            section_path=c.section_path,
            snippet=c.snippet,
            score=-scores.get(i, 0.0),
        )
        for i, c in indexed[:limit]
    ]


# ── rerankers ────────────────────────────────────────────────────────


class BaseReranker:
    kind: str = "none"
    provider: str = "none"

    def __init__(self, *, max_chars: int = RERANK_MAX_CHARS, fail_mode: str = "soft") -> None:
        self.max_chars = max_chars
        self.fail_mode = fail_mode

    @property
    def available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "provider": self.provider}

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        body_lookup: BodyLookup,
        *,
        limit: int = 10,
    ) -> RerankOutcome:
        candidates = list(candidates)
        if not candidates:
            return RerankOutcome([], applied=False, kind=self.kind)
        items = _passages(candidates, body_lookup, self.max_chars)
        try:
            scores = self._score(query, items)
        except RerankError as exc:
            if self.fail_mode == "hard":
                raise
            logger.warning("rerank (%s/%s) failed: %s; returning fused order",
                           self.kind, self.provider, exc)
            return RerankOutcome(candidates[:limit], applied=False, kind=self.kind, error=str(exc))
        if not scores:
            msg = "reranker returned no scores"
            if self.fail_mode == "hard":
                raise RerankError(msg)
            logger.warning("rerank: %s; returning fused order", msg)
            return RerankOutcome(candidates[:limit], applied=False, kind=self.kind, error=msg)
        return RerankOutcome(_apply_scores(candidates, scores, limit), applied=True, kind=self.kind)

    def _score(self, query: str, items: list[dict[str, Any]]) -> dict[int, float]:  # pragma: no cover
        raise NotImplementedError


class NoReranker(BaseReranker):
    kind = "none"

    @property
    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str | None:
        return "rerank is disabled on this server (AMPERSTAND_RERANK_KIND=none)"

    def rerank(self, query, candidates, body_lookup, *, limit=10) -> RerankOutcome:
        return RerankOutcome(list(candidates)[:limit], applied=False, kind="none",
                             error=self.unavailable_reason())


class LLMReranker(BaseReranker):
    """The original grader: one JSON-mode chat completion scoring every
    passage 0-10. Works against OpenAI or any OpenAI-compatible chat host."""

    kind = "llm"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str | None = None,
        provider: str = "openai",
        timeout_s: float = 20.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = provider
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def unavailable_reason(self) -> str | None:
        if not self.api_key:
            return ("Rerank was requested but is disabled. Set OPENAI_API_KEY "
                    "(or AMPERSTAND_RERANK_API_KEY) on the server to enable.")
        return None

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "provider": self.provider, "model": self.model}

    def _score(self, query: str, items: list[dict[str, Any]]) -> dict[int, float]:
        if not self.api_key:
            raise RerankError("OPENAI_API_KEY not set")
        prompt = (
            f"You are a search relevance judge. Rate each passage on how well "
            f"it answers the query, 0-10 (10 = directly answers, 5 = related, "
            f"0 = unrelated). Be strict: most passages should score below 5.\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{json.dumps(items, ensure_ascii=False)}\n\n"
            f"Return JSON: {{\"scores\": [{{\"id\": <int>, \"score\": <int>}}, ...]}}"
        )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RerankError("openai package not installed") from exc
        try:
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout_s}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"LLM call failed: {exc}") from exc
        return {
            int(s["id"]): float(s["score"])
            for s in parsed.get("scores", [])
            if isinstance(s, dict) and "id" in s and "score" in s
        }


class HTTPCrossEncoderReranker(BaseReranker):
    """A cross-encoder behind an HTTP `/rerank` endpoint.

    Request/response shapes by provider (verified Oct 2026):
      tei       POST {base}/rerank {query, texts, raw_scores:false, truncate:true}
                → [{index, score}]
      infinity  POST {base}/rerank {query, documents, model?, top_n?}
                → {results:[{index, relevance_score}]}
      cohere    POST {base}/rerank {model, query, documents, top_n}      (base = .../v2)
                → {results:[{index, relevance_score}]}
      jina      POST {base}/rerank {model, query, documents, top_n}      (base = .../v1)
                → {results:[{index, relevance_score}]}
      voyage    POST {base}/rerank {model, query, documents, top_k}      (base = .../v1)
                → {data:[{index, relevance_score}]}
    """

    kind = "cross_encoder"
    _DEFAULT_URLS = {"cohere": COHERE_RERANK_URL, "jina": JINA_RERANK_URL, "voyage": VOYAGE_RERANK_URL}

    def __init__(
        self,
        *,
        provider: str,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if provider not in ("tei", "infinity", "cohere", "jina", "voyage"):
            raise RerankError(f"unknown cross-encoder provider {provider!r}")
        self.provider = provider
        base_url = base_url or self._DEFAULT_URLS.get(provider)
        if not base_url:
            raise RerankError(f"AMPERSTAND_RERANK_BASE_URL is required for provider={provider}")
        if provider in ("cohere", "jina", "voyage") and not api_key:
            raise RerankError(f"AMPERSTAND_RERANK_API_KEY is required for provider={provider}")
        if provider in ("cohere", "jina", "voyage") and not model:
            raise RerankError(f"AMPERSTAND_RERANK_MODEL is required for provider={provider}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(headers=headers, timeout=timeout_s, transport=transport)

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "provider": self.provider, "model": self.model,
                "base_url": self.base_url}

    def _body(self, query: str, docs: list[str]) -> dict[str, Any]:
        if self.provider == "tei":
            return {"query": query, "texts": docs, "raw_scores": False, "truncate": True}
        body: dict[str, Any] = {"query": query, "documents": docs}
        if self.model:
            body["model"] = self.model
        if self.provider == "voyage":
            body["top_k"] = len(docs)
        else:
            body["top_n"] = len(docs)
        return body

    def _score(self, query: str, items: list[dict[str, Any]]) -> dict[int, float]:
        docs = [_passage_text(it) for it in items]
        try:
            resp = self._client.post(f"{self.base_url}/rerank", json=self._body(query, docs))
        except httpx.HTTPError as exc:
            raise RerankError(f"{self.provider} rerank request failed: {exc}") from exc
        if resp.status_code >= 300:
            raise RerankError(f"{self.provider} rerank HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            payload = resp.json()
        except ValueError as exc:
            raise RerankError(f"{self.provider} rerank returned non-JSON") from exc
        rows: Any
        if isinstance(payload, list):                       # TEI
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("results") or payload.get("data") or []
        else:
            rows = []
        scores: dict[int, float] = {}
        for r in rows:
            if not isinstance(r, dict) or "index" not in r:
                continue
            val = r.get("relevance_score", r.get("score"))
            if val is None:
                continue
            scores[int(r["index"])] = float(val)
        return scores


class ONNXCrossEncoderReranker(BaseReranker):
    """In-process cross-encoder (optional extra: amperstand-core[onnx]).

    Tested with cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    (`onnx/model_quint8_avx2.onnx`): query/passage pairs → one logit each.
    """

    kind = "cross_encoder"
    provider = "onnx"

    def __init__(
        self,
        *,
        model_dir: str,
        onnx_file: str = "onnx/model.onnx",
        threads: int = 1,
        max_tokens: int = 512,
        batch_size: int = 8,
        model: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        try:
            import numpy as np  # noqa: F401
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RerankError(
                "onnx reranker needs onnxruntime+tokenizers+numpy; "
                "pip install 'amperstand-core[onnx]'"
            ) from exc
        from pathlib import Path

        d = Path(model_dir)
        if not (d / "tokenizer.json").exists() or not (d / onnx_file).exists():
            raise RerankError(f"onnx reranker: missing tokenizer.json or {onnx_file} in {d}")
        self.model = model or d.name
        self.batch_size = max(1, batch_size)
        self._tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tok.enable_truncation(max_tokens)
        self._tok.no_padding()
        pad = self._tok.token_to_id("<pad>")
        self._pad_id = pad if pad is not None else 0
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(d / onnx_file), so, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "provider": self.provider, "model": self.model}

    def _score(self, query: str, items: list[dict[str, Any]]) -> dict[int, float]:
        import numpy as np

        docs = [_passage_text(it) for it in items]
        scores: dict[int, float] = {}
        try:
            for start in range(0, len(docs), self.batch_size):
                chunk = docs[start : start + self.batch_size]
                encs = self._tok.encode_batch([(query, d) for d in chunk])
                L = max(len(e.ids) for e in encs)
                ids = np.full((len(encs), L), self._pad_id, dtype=np.int64)
                mask = np.zeros((len(encs), L), dtype=np.int64)
                for r, e in enumerate(encs):
                    ids[r, : len(e.ids)] = e.ids
                    mask[r, : len(e.ids)] = 1
                feeds = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in self._inputs:
                    feeds["token_type_ids"] = np.zeros_like(ids)
                logits = self._sess.run(None, feeds)[0]
                for r in range(len(chunk)):
                    scores[start + r] = float(np.asarray(logits[r]).reshape(-1)[0])
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"onnx reranker failed: {exc}") from exc
        return scores


# ── config + factory ─────────────────────────────────────────────────


@dataclass
class RerankConfig:
    kind: str = "llm"
    provider: str = "openai"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_chars: int = RERANK_MAX_CHARS
    timeout_s: float = 20.0
    fail_mode: str = "soft"
    onnx_dir: str | None = None
    onnx_file: str = "onnx/model.onnx"
    onnx_threads: int = 1
    onnx_max_tokens: int = 512
    extra: dict[str, str] = field(default_factory=dict)


_RERANK_KEY_FALLBACK = {
    "openai": "OPENAI_API_KEY", "openai_compatible": "OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY", "jina": "JINA_API_KEY", "voyage": "VOYAGE_API_KEY",
}


def rerank_config_from_env(env: Mapping[str, str] | None = None) -> RerankConfig:
    e = os.environ if env is None else env
    g = lambda k, d=None: (e.get(k) or "").strip() or d  # noqa: E731
    kind = (g("AMPERSTAND_RERANK_KIND", "llm") or "llm").lower().replace("-", "_")
    if kind not in KINDS:
        raise RerankError(f"AMPERSTAND_RERANK_KIND={kind!r} is not one of {', '.join(KINDS)}")
    provider = g("AMPERSTAND_RERANK_PROVIDER")
    if not provider:
        provider = "openai" if kind == "llm" else ("tei" if kind == "cross_encoder" else "none")
    provider = provider.lower().replace("-", "_")
    if kind == "llm" and provider not in LLM_PROVIDERS:
        raise RerankError(f"rerank kind=llm needs provider in {LLM_PROVIDERS}, got {provider!r}")
    if kind == "cross_encoder" and provider not in CROSS_ENCODER_PROVIDERS:
        raise RerankError(
            f"rerank kind=cross_encoder needs provider in {CROSS_ENCODER_PROVIDERS}, got {provider!r}"
        )
    model = g("AMPERSTAND_RERANK_MODEL") or (DEFAULT_LLM_MODEL if kind == "llm" else None)
    api_key = g("AMPERSTAND_RERANK_API_KEY") or (
        g(_RERANK_KEY_FALLBACK[provider]) if provider in _RERANK_KEY_FALLBACK else None
    )
    base_url = g("AMPERSTAND_RERANK_BASE_URL")
    if not base_url and provider == "openai":
        base_url = g("OPENAI_BASE_URL")
    fail_mode = (g("AMPERSTAND_RERANK_FAIL_MODE", "soft") or "soft").lower()
    if fail_mode not in ("soft", "hard"):
        raise RerankError("AMPERSTAND_RERANK_FAIL_MODE must be soft|hard")
    return RerankConfig(
        kind=kind, provider=provider, model=model, base_url=base_url, api_key=api_key,
        max_chars=int(g("AMPERSTAND_RERANK_MAX_CHARS", str(RERANK_MAX_CHARS)) or RERANK_MAX_CHARS),
        timeout_s=float(g("AMPERSTAND_RERANK_TIMEOUT_S", "20") or 20),
        fail_mode=fail_mode,
        onnx_dir=g("AMPERSTAND_RERANK_ONNX_DIR"),
        onnx_file=g("AMPERSTAND_RERANK_ONNX_FILE", "onnx/model.onnx") or "onnx/model.onnx",
        onnx_threads=int(g("AMPERSTAND_RERANK_ONNX_THREADS", "1") or 1),
        onnx_max_tokens=int(g("AMPERSTAND_RERANK_ONNX_MAX_TOKENS", "512") or 512),
    )


def build_reranker(cfg: RerankConfig, **overrides) -> BaseReranker:
    common = dict(max_chars=cfg.max_chars, fail_mode=cfg.fail_mode)
    if cfg.kind == "none" or not rerank_enabled():
        return NoReranker(**common)
    if cfg.kind == "llm":
        return LLMReranker(
            api_key=cfg.api_key, model=cfg.model or DEFAULT_LLM_MODEL, base_url=cfg.base_url,
            provider=cfg.provider, timeout_s=cfg.timeout_s, **common,
        )
    if cfg.provider == "onnx":
        if not cfg.onnx_dir:
            raise RerankError("AMPERSTAND_RERANK_ONNX_DIR is required for provider=onnx")
        return ONNXCrossEncoderReranker(
            model_dir=cfg.onnx_dir, onnx_file=cfg.onnx_file, threads=cfg.onnx_threads,
            max_tokens=cfg.onnx_max_tokens, model=cfg.model, **common,
        )
    return HTTPCrossEncoderReranker(
        provider=cfg.provider, base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model,
        timeout_s=cfg.timeout_s, **common, **overrides,
    )


def build_reranker_from_env(env: Mapping[str, str] | None = None, **overrides) -> BaseReranker:
    return build_reranker(rerank_config_from_env(env), **overrides)


# ── back-compat wrapper ──────────────────────────────────────────────


def rerank_with_llm(
    query: str,
    candidates: Sequence[SearchResult],
    body_lookup: BodyLookup,
    *,
    limit: int = 10,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> list[SearchResult]:
    """Re-rank `candidates` by LLM-judged relevance to `query`.

    Kept for callers of the pre-provider API. On any LLM error returns the
    first `limit` candidates in their original (RRF) order — never raises.
    New code should use `build_reranker_from_env().rerank(...)`, which also
    reports whether the rerank was applied.
    """
    if not candidates:
        return []
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("rerank: OPENAI_API_KEY not set; falling back to RRF order")
        return list(candidates)[:limit]
    r = LLMReranker(api_key=api_key, model=model or RERANK_MODEL, base_url=base_url, fail_mode="soft")
    return r.rerank(query, candidates, body_lookup, limit=limit).results
