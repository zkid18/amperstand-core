"""Env-driven configuration and factory for embedding providers.

    AMPERSTAND_EMBED_PROVIDER   openai (default) | openai_compatible | voyage | jina |
                                cohere | tei | onnx | none
    AMPERSTAND_EMBED_MODEL      model name sent to the API / recorded in the index
    AMPERSTAND_EMBED_DIM        vector dimension the index is built with (default 512)
    AMPERSTAND_EMBED_BASE_URL   host for openai_compatible / tei (required), or an
                                override for openai / voyage / jina / cohere
    AMPERSTAND_EMBED_API_KEY    key; falls back to OPENAI_API_KEY / VOYAGE_API_KEY /
                                JINA_API_KEY / COHERE_API_KEY per provider
    AMPERSTAND_EMBED_REQUEST_DIM   auto (default) | 1 | 0 — ask the API for DIM-sized
                                vectors (OpenAI dimensions=, Voyage/Cohere
                                output_dimension=, Jina dimensions=). auto = on for
                                openai, off elsewhere.
    AMPERSTAND_EMBED_MATRYOSHKA  1 to slice longer vectors to DIM client-side (+renorm)
    AMPERSTAND_EMBED_QUERY_PREFIX / AMPERSTAND_EMBED_DOC_PREFIX   e.g. "query: " / "passage: "
    AMPERSTAND_EMBED_MAX_TOKENS  truncation budget per text (default 8000)
    AMPERSTAND_EMBED_BATCH_SIZE  texts per request (default 100; Cohere caps at 96)
    AMPERSTAND_EMBED_TIMEOUT_S   HTTP timeout (default 30)
    AMPERSTAND_EMBED_ONNX_DIR / _ONNX_FILE / _ONNX_POOLING / _ONNX_THREADS / _ONNX_MAX_TOKENS
                                in-process ONNX (provider=onnx) — optional extra

With nothing set, the result is exactly the pre-existing behaviour: the
OpenAI SDK, text-embedding-3-small, dimensions=512, keyed by OPENAI_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from amperstand_core.embeddings.base import (
    BATCH_SIZE,
    DEFAULT_MODEL,
    EMBED_DIM,
    MAX_TOKENS,
    BaseEmbedder,
    EmbeddingConfigError,
)

PROVIDERS = ("openai", "openai_compatible", "voyage", "jina", "cohere", "tei", "onnx", "none")

_KEY_FALLBACK = {
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "jina": "JINA_API_KEY",
    "cohere": "COHERE_API_KEY",
}


def _truthy(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class EmbedConfig:
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    dim: int = EMBED_DIM
    base_url: str | None = None
    api_key: str | None = None
    request_dim: bool = True
    matryoshka: bool = False
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int = MAX_TOKENS
    batch_size: int = BATCH_SIZE
    timeout_s: float = 30.0
    onnx_dir: str | None = None
    onnx_file: str = "onnx/model.onnx"
    onnx_pooling: str = "cls"
    onnx_threads: int = 1
    onnx_max_tokens: int = 2048
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def describe(self) -> dict[str, object]:
        """Safe to log: no key."""
        return {
            "provider": self.provider, "model": self.model, "dim": self.dim,
            "base_url": self.base_url, "api_key_set": bool(self.api_key),
            "request_dim": self.request_dim, "matryoshka": self.matryoshka,
        }


def embed_config_from_env(env: Mapping[str, str] | None = None) -> EmbedConfig:
    e = os.environ if env is None else env
    g = lambda k, d=None: (e.get(k) or "").strip() or d  # noqa: E731

    provider = (g("AMPERSTAND_EMBED_PROVIDER", "openai") or "openai").lower().replace("-", "_")
    if provider not in PROVIDERS:
        raise EmbeddingConfigError(
            f"AMPERSTAND_EMBED_PROVIDER={provider!r} is not one of {', '.join(PROVIDERS)}"
        )
    model = g("AMPERSTAND_EMBED_MODEL") or (DEFAULT_MODEL if provider == "openai" else "")
    try:
        dim = int(g("AMPERSTAND_EMBED_DIM", str(EMBED_DIM)))
    except ValueError as exc:
        raise EmbeddingConfigError("AMPERSTAND_EMBED_DIM must be an integer") from exc

    api_key = g("AMPERSTAND_EMBED_API_KEY")
    if not api_key and provider in _KEY_FALLBACK:
        api_key = g(_KEY_FALLBACK[provider])
    base_url = g("AMPERSTAND_EMBED_BASE_URL")
    if not base_url and provider == "openai":
        base_url = g("OPENAI_BASE_URL")  # the SDK honoured this already

    raw_req = (g("AMPERSTAND_EMBED_REQUEST_DIM", "auto") or "auto").lower()
    request_dim = provider == "openai" if raw_req == "auto" else _truthy(raw_req, False)

    def _int(k: str, d: int) -> int:
        try:
            return int(g(k, str(d)))
        except ValueError as exc:
            raise EmbeddingConfigError(f"{k} must be an integer") from exc

    return EmbedConfig(
        provider=provider,
        model=model,
        dim=dim,
        base_url=base_url,
        api_key=api_key,
        request_dim=request_dim,
        matryoshka=_truthy(g("AMPERSTAND_EMBED_MATRYOSHKA"), False),
        query_prefix=e.get("AMPERSTAND_EMBED_QUERY_PREFIX", "") or "",
        doc_prefix=e.get("AMPERSTAND_EMBED_DOC_PREFIX", "") or "",
        max_tokens=_int("AMPERSTAND_EMBED_MAX_TOKENS", MAX_TOKENS),
        batch_size=_int("AMPERSTAND_EMBED_BATCH_SIZE", BATCH_SIZE),
        timeout_s=float(g("AMPERSTAND_EMBED_TIMEOUT_S", "30") or 30),
        onnx_dir=g("AMPERSTAND_EMBED_ONNX_DIR"),
        onnx_file=g("AMPERSTAND_EMBED_ONNX_FILE", "onnx/model.onnx") or "onnx/model.onnx",
        onnx_pooling=(g("AMPERSTAND_EMBED_ONNX_POOLING", "cls") or "cls").lower(),
        onnx_threads=_int("AMPERSTAND_EMBED_ONNX_THREADS", 1),
        onnx_max_tokens=_int("AMPERSTAND_EMBED_ONNX_MAX_TOKENS", 2048),
    )


def build_embedder(cfg: EmbedConfig, **overrides) -> BaseEmbedder | None:
    """Instantiate the configured provider. Returns None for provider=none.

    Raises `EmbeddingConfigError` when the configuration is incomplete
    (missing key / base URL / model dir) — callers decide whether that
    disables semantic search (server) or aborts (admin CLI).
    `overrides` are passed to the constructor (tests inject an httpx
    transport this way).
    """
    if not cfg.enabled:
        return None
    common = dict(
        dim=cfg.dim,
        max_tokens=cfg.max_tokens,
        batch_size=cfg.batch_size,
        query_prefix=cfg.query_prefix,
        doc_prefix=cfg.doc_prefix,
        matryoshka=cfg.matryoshka,
    )
    p = cfg.provider
    if p == "openai":
        if not cfg.api_key:
            raise EmbeddingConfigError(
                "OPENAI_API_KEY is not set. Embedding-based search is disabled."
            )
        from amperstand_core.embeddings.openai_provider import OpenAIEmbedder
        return OpenAIEmbedder(
            api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url,
            request_dim=cfg.request_dim, timeout_s=cfg.timeout_s, **common, **overrides,
        )
    if p == "onnx":
        if not cfg.onnx_dir:
            raise EmbeddingConfigError("AMPERSTAND_EMBED_ONNX_DIR is required for provider=onnx")
        from amperstand_core.embeddings.onnx_provider import ONNXEmbedder
        return ONNXEmbedder(
            model_dir=cfg.onnx_dir, onnx_file=cfg.onnx_file, model=cfg.model or None,
            pooling=cfg.onnx_pooling, threads=cfg.onnx_threads,
            max_model_tokens=cfg.onnx_max_tokens, **common, **overrides,
        )

    from amperstand_core.embeddings import http_provider as hp

    cls = {
        "openai_compatible": hp.OpenAICompatibleEmbedder,
        "tei": hp.TEIEmbedder,
        "voyage": hp.VoyageEmbedder,
        "jina": hp.JinaEmbedder,
        "cohere": hp.CohereEmbedder,
    }[p]
    if p in ("openai_compatible", "tei") and not cfg.base_url:
        raise EmbeddingConfigError(f"AMPERSTAND_EMBED_BASE_URL is required for provider={p}")
    if p in ("openai_compatible", "voyage", "jina", "cohere") and not cfg.model:
        raise EmbeddingConfigError(f"AMPERSTAND_EMBED_MODEL is required for provider={p}")
    if p in ("voyage", "jina", "cohere") and not cfg.api_key:
        raise EmbeddingConfigError(
            f"AMPERSTAND_EMBED_API_KEY (or {_KEY_FALLBACK[p]}) is required for provider={p}"
        )
    return cls(
        model=cfg.model or "tei", base_url=cfg.base_url, api_key=cfg.api_key,
        request_dim=cfg.request_dim, timeout_s=cfg.timeout_s, **common, **overrides,
    )


def build_embedder_from_env(env: Mapping[str, str] | None = None, **overrides) -> BaseEmbedder | None:
    return build_embedder(embed_config_from_env(env), **overrides)
