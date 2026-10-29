"""Provider-neutral embedding contract.

Every embedding provider (OpenAI, an OpenAI-compatible host, Voyage, Jina,
Cohere, a TEI/Infinity sidecar, an in-process ONNX model) implements
`BaseEmbedder`. The vector index only ever sees this interface plus an
`EmbedSignature` naming the provider/model/dim the vectors came from — the
index stores that signature and refuses to mix vectors from different
models.

Shared behaviour lives here so every provider gets it for free:

- truncation to a token budget (tiktoken when available, else a char cap),
- query/document prefixes (e5-style "query: ", EmbeddingGemma prompts),
- batching, and
- the dimension guard: a vector that doesn't match the configured dim is
  an error unless Matryoshka slicing was explicitly enabled.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
# 512 instead of the model's native 1536 — text-embedding-3-small uses
# Matryoshka Representation Learning so a 512-dim slice is still a
# well-normalized vector with most of the retrieval quality. Trade-off:
# brute-force KNN over our ~43k section vectors drops from ~6s to ~0.1s
# on a 1 vCPU droplet, which is the difference between "research tool"
# and "interactive search". Recall loss is negligible at this scale.
EMBED_DIM = 512
# OpenAI's hard limit is 8192 tokens; leave a small headroom for any
# tokenizer drift. We truncate by token count when tiktoken is available
# and fall back to a tight char-based budget when it isn't.
MAX_TOKENS = 8000
CHAR_FALLBACK = 16_000  # ~4000 tokens worst case at 2 chars/token (CJK)
BATCH_SIZE = 100
MAX_RETRIES = 4

KIND_QUERY = "query"
KIND_DOCUMENT = "document"

# tiktoken is optional. If it imports, we truncate by exact token count.
# If not, we fall back to a conservative char limit. Either way the
# request stays under OpenAI's 8192-token cap.
try:
    import tiktoken
    _ENCODER = tiktoken.encoding_for_model(DEFAULT_MODEL)
except Exception:  # noqa: BLE001
    _ENCODER = None


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Cap `text` at `max_tokens` (cl100k tokens when tiktoken is around).

    Non-OpenAI tokenizers count differently, so for those providers this
    is a *budget*, not an exact cap: TEI/ONNX providers additionally ask
    the model-side tokenizer to truncate.
    """
    if _ENCODER is not None:
        tokens = _ENCODER.encode(text, disallowed_special=())
        if len(tokens) <= max_tokens:
            return text
        return _ENCODER.decode(tokens[:max_tokens])
    # ~4 chars/token for Latin text, 2 for CJK; scale the char cap with the
    # token budget so a smaller budget still bites.
    char_cap = min(CHAR_FALLBACK, max_tokens * 2)
    if len(text) <= char_cap:
        return text
    return text[:char_cap]


# Kept for back-compat with callers that imported the private helper.
_truncate = truncate_text


@dataclass
class EmbeddingResult:
    text_hash: str
    vector: list[float]


class EmbeddingError(RuntimeError):
    """Any failure to produce embeddings: provider down, quota, bad config."""


class EmbeddingConfigError(EmbeddingError):
    """The provider could not even be constructed from the configuration."""


@dataclass(frozen=True)
class EmbedSignature:
    """Identity of the vector space an index was built in.

    Two indexes are compatible only if all three match. `provider` is the
    *kind* of provider (openai, voyage, tei, onnx, ...), not the host: the
    same model served by TEI on another box produces the same vectors.
    `model` is whatever the operator configured — for ONNX that is the
    model directory's basename, for hosted APIs the API model name.
    """

    provider: str
    model: str
    dim: int

    def as_dict(self) -> dict[str, str | int]:
        return {"provider": self.provider, "model": self.model, "dim": self.dim}

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}@{self.dim}"


def classify_failure(status: int | None, message: str) -> str:
    """Decide whether a provider error is worth retrying.

    Returns "fatal" for errors a retry cannot fix (bad request, auth,
    exhausted quota/credit balance) and "retry" for everything else
    (rate limits, 5xx, timeouts, connection resets).

    The quota case is the one that bit production: OpenAI answers an
    empty balance with 429 `insufficient_quota`, which the old retry loop
    treated like a rate limit — four attempts, ~7s of backoff, then a
    bare 500. Failing fast turns that into an immediate, explicit 503.
    """
    low = message.lower()
    if status in (400, 401, 403, 404, 413, 422):
        return "fatal"
    if status == 429 and any(
        s in low for s in ("insufficient_quota", "credit_balance", "no credits", "billing")
    ):
        return "fatal"
    return "retry"


class BaseEmbedder:
    """Contract every provider implements.

    Subclasses implement `_embed_raw(texts, kind)` for one already-truncated,
    already-prefixed chunk of at most `batch_size` texts, raising
    `EmbeddingError` on failure. `kind` is "query" or "document" so APIs
    with asymmetric input types (Voyage, Cohere, Jina) can pass it on.
    Everything else — prefixes, truncation, batching, the dim guard — is
    handled here.
    """

    provider: str = "base"

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        max_tokens: int = MAX_TOKENS,
        batch_size: int = BATCH_SIZE,
        query_prefix: str = "",
        doc_prefix: str = "",
        matryoshka: bool = False,
    ) -> None:
        if dim <= 0:
            raise EmbeddingConfigError(f"embedding dim must be positive, got {dim}")
        self.model = model
        self.dim = dim
        self.max_tokens = max_tokens
        self.batch_size = max(1, batch_size)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.matryoshka = matryoshka

    # ── public API ───────────────────────────────────────────────────

    @property
    def signature(self) -> EmbedSignature:
        return EmbedSignature(provider=self.provider, model=self.model, dim=self.dim)

    def embed(self, text: str) -> list[float]:
        """Embed one *query*. Applies the query prefix."""
        return self._embed_kind([text], kind=KIND_QUERY)[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *documents* (sections). Applies the document prefix and
        splits into `batch_size` chunks."""
        return self._embed_kind(texts, kind=KIND_DOCUMENT)

    def describe(self) -> dict[str, object]:
        """Operator-facing summary (never includes secrets)."""
        return {"provider": self.provider, "model": self.model, "dim": self.dim}

    # ── provider hook ────────────────────────────────────────────────

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    # ── internals ────────────────────────────────────────────────────

    def _embed_kind(self, texts: Sequence[str], *, kind: str) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.query_prefix if kind == KIND_QUERY else self.doc_prefix
        prepared = [
            prefix + truncate_text(t, self.max_tokens) if t else (prefix + " " if prefix else " ")
            for t in texts
        ]
        out: list[list[float]] = []
        for i in range(0, len(prepared), self.batch_size):
            chunk = prepared[i : i + self.batch_size]
            vectors = self._embed_raw(chunk, kind=kind)
            if len(vectors) != len(chunk):
                raise EmbeddingError(
                    f"{self.provider}: asked for {len(chunk)} embeddings, got {len(vectors)}"
                )
            out.extend(self._fit(v) for v in vectors)
        return out

    def _fit(self, vector: Sequence[float]) -> list[float]:
        """Enforce the configured dimension.

        Exact match → pass through. Longer vector + matryoshka enabled →
        slice and re-normalise (valid for MRL-trained models such as
        text-embedding-3, granite-311m-r2, EmbeddingGemma, jina-v3).
        Anything else is a configuration error, not something to paper over.
        """
        n = len(vector)
        if n == self.dim:
            return [float(x) for x in vector]
        if n > self.dim and self.matryoshka:
            head = [float(x) for x in vector[: self.dim]]
            norm = math.sqrt(sum(x * x for x in head))
            return [x / norm for x in head] if norm > 0 else head
        raise EmbeddingError(
            f"{self.provider}/{self.model} returned a {n}-dim vector but the index "
            f"is configured for {self.dim} dims"
            + ("" if n <= self.dim else " (set AMPERSTAND_EMBED_MATRYOSHKA=1 to slice MRL models)")
        )
