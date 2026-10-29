"""OpenAI embeddings through the official SDK — the default provider.

This is the pre-existing code path, kept behaviourally identical for the
default configuration: `text-embedding-3-small`, `dimensions=512`, batches
of 100, four attempts with exponential backoff, tiktoken truncation at
8000 tokens. Two additions:

- `base_url` is accepted explicitly (the SDK already honoured
  OPENAI_BASE_URL; now the operator can set it per provider), and
- a 429 that says the balance is exhausted (`insufficient_quota`) is not
  retried. Retrying it only delayed the inevitable by ~7s in production.
"""

from __future__ import annotations

import logging
import os
import time

from amperstand_core.embeddings.base import (
    DEFAULT_MODEL,
    EMBED_DIM,
    MAX_RETRIES,
    BaseEmbedder,
    EmbeddingError,
    classify_failure,
)

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Thin OpenAI embeddings wrapper. Construct via `from_env()` or the
    provider factory (`amperstand_core.embeddings.build_embedder_from_env`)."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int = EMBED_DIM,
        base_url: str | None = None,
        request_dim: bool = True,
        timeout_s: float | None = None,
        **kwargs,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "openai package not installed. Run: pip install openai"
            ) from exc
        super().__init__(model=model, dim=dim, **kwargs)
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout_s:
            client_kwargs["timeout"] = timeout_s
        # Leave the SDK's own retries off: we run the retry loop ourselves
        # so the quota classification applies on the first failure.
        client_kwargs["max_retries"] = 0
        self._client = OpenAI(**client_kwargs)
        self.request_dim = request_dim

    @classmethod
    def from_env(cls, *, model: str = DEFAULT_MODEL) -> "OpenAIEmbedder":
        """Back-compat constructor: OPENAI_API_KEY (+ optional OPENAI_BASE_URL)."""
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise EmbeddingError(
                "OPENAI_API_KEY is not set. Embedding-based search is disabled."
            )
        return cls(api_key=key, model=model, base_url=os.environ.get("OPENAI_BASE_URL") or None)

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        delay = 1.0
        for attempt in range(MAX_RETRIES):
            try:
                params: dict = {"input": texts, "model": self.model}
                if self.request_dim:
                    params["dimensions"] = self.dim
                resp = self._client.embeddings.create(**params)
                data = sorted(resp.data, key=lambda d: d.index)
                return [list(d.embedding) for d in data]
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
                # 400 = bad input (oversize, malformed); 401/403 = key;
                # 429 insufficient_quota = empty balance. Retrying won't fix
                # any of these — re-raise immediately with the cause.
                if classify_failure(status, str(exc)) == "fatal":
                    raise EmbeddingError(
                        f"openai embeddings rejected ({status}): {exc}"
                    ) from exc
                if attempt == MAX_RETRIES - 1:
                    raise EmbeddingError(f"embeddings failed after retries: {exc}") from exc
                logger.warning(
                    "embedding attempt %d failed (%s); retrying in %.1fs",
                    attempt + 1, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
        # Unreachable
        raise EmbeddingError("embedding retry loop exited without result")
