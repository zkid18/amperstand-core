"""Search module — section-aware FTS5 + vector indices over the markdown vault."""

from amperstand_core.search.errors import SearchError
from amperstand_core.search.index import SearchIndex
from amperstand_core.search.indexer import SearchIndexer
from amperstand_core.search.models import Section, SearchResult
from amperstand_core.search.parser import parse_sections
from amperstand_core.search.rerank import (
    BaseReranker,
    RerankError,
    RerankOutcome,
    build_reranker_from_env,
    rerank_enabled,
    rerank_with_llm,
)
from amperstand_core.search.vec_index import IndexSignatureMismatch, VectorIndex
from amperstand_core.search.vec_indexer import VectorIndexer

__all__ = [
    "SearchIndex",
    "SearchIndexer",
    "SearchResult",
    "Section",
    "SearchError",
    "BaseReranker",
    "RerankError",
    "RerankOutcome",
    "IndexSignatureMismatch",
    "build_reranker_from_env",
    "VectorIndex",
    "VectorIndexer",
    "parse_sections",
    "rerank_enabled",
    "rerank_with_llm",
]
