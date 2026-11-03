"""Embedding providers for the vector index.

Back-compat surface (unchanged for existing callers):

    from amperstand_core.embeddings import Embedder, EMBED_DIM, EmbeddingError
    Embedder.from_env()            # OpenAI, keyed by OPENAI_API_KEY

New surface:

    from amperstand_core.embeddings import build_embedder_from_env, EmbedSignature
    embedder = build_embedder_from_env()   # picks the provider from AMPERSTAND_EMBED_*
    embedder.signature                     # provider/model/dim recorded in vec_meta_kv
"""

from amperstand_core.embeddings.base import (
    BATCH_SIZE,
    DEFAULT_MODEL,
    EMBED_DIM,
    MAX_RETRIES,
    MAX_TOKENS,
    BaseEmbedder,
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingResult,
    EmbedSignature,
    _truncate,
    classify_failure,
    truncate_text,
)
from amperstand_core.embeddings.config import (
    PROVIDERS,
    EmbedConfig,
    build_embedder,
    build_embedder_from_env,
    embed_config_from_env,
)
from amperstand_core.embeddings.openai_provider import OpenAIEmbedder

# The historical name. Existing code does `Embedder.from_env()` and
# `isinstance(x, Embedder)` — both keep working.
Embedder = OpenAIEmbedder

# The signature production's vec.db was built with before signatures were
# recorded at all (schema_version=2, OpenAI text-embedding-3-small @512).
LEGACY_SIGNATURE = EmbedSignature(provider="openai", model=DEFAULT_MODEL, dim=EMBED_DIM)

__all__ = [
    "BATCH_SIZE",
    "DEFAULT_MODEL",
    "EMBED_DIM",
    "LEGACY_SIGNATURE",
    "MAX_RETRIES",
    "MAX_TOKENS",
    "PROVIDERS",
    "BaseEmbedder",
    "EmbedConfig",
    "EmbedSignature",
    "Embedder",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingResult",
    "OpenAIEmbedder",
    "_truncate",
    "build_embedder",
    "build_embedder_from_env",
    "classify_failure",
    "embed_config_from_env",
    "truncate_text",
]
