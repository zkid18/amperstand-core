"""Build the (store, FTS indexer, vec indexer, reranker) graph and bootstrap them.

The MarkdownStore takes a single on_change hook, so we fan out internally:
each indexer (FTS, vec) sees every event but failures in one don't block
the other. Vec is optional — if no embedding provider is configured (by
default: OPENAI_API_KEY unset) or sqlite-vec isn't loadable, the rest of
the system runs unchanged and semantic search just isn't available.

Providers are chosen from the environment (`AMPERSTAND_EMBED_*`,
`AMPERSTAND_RERANK_*`); with nothing set this is the pre-provider
behaviour: OpenAI embeddings + the gpt-4o-mini LLM grader.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from amperstand_core.embeddings import (
    EmbeddingConfigError,
    build_embedder,
    embed_config_from_env,
)
from amperstand_core.search import (
    BaseReranker,
    RerankError,
    SearchIndex,
    SearchIndexer,
    VectorIndex,
    VectorIndexer,
    build_reranker_from_env,
)
from amperstand_core.search.rerank import NoReranker
from amperstand_core.store import MarkdownStore

DEFAULT_DATA_DIR = Path.home() / ".amperstand" / "vault"

log = logging.getLogger(__name__)


def _resolve_data_dir() -> Path:
    raw = os.environ.get("AMPERSTAND_DATA_DIR")
    return Path(raw).expanduser() if raw else DEFAULT_DATA_DIR


@dataclass
class Indexers:
    store: MarkdownStore
    fts: SearchIndexer
    vec: VectorIndexer | None  # None when no embedding provider / sqlite-vec missing
    vec_reason: str | None = None  # why vec is None, for the 503 detail


@lru_cache(maxsize=1)
def get_indexers() -> Indexers:
    root = _resolve_data_dir()
    fts_index = SearchIndex(root / ".store" / "search.db")

    def _on_change(evt):
        # FTS first — keyword search is the baseline. Vec is best-effort.
        bundle = get_indexers()
        try:
            bundle.fts.handle_change(evt)
        except Exception:
            log.exception("FTS indexer failed handling change")
        if bundle.vec is not None:
            try:
                bundle.vec.handle_change(evt)
            except Exception:
                log.exception("vec indexer failed handling change")

    store = MarkdownStore(root, on_change=_on_change)
    fts_indexer = SearchIndexer(store=store, index=fts_index)

    fts_added = fts_indexer.bootstrap(force=False)
    if fts_added:
        log.info("FTS bootstrap reindexed %d docs", fts_added)

    vec_indexer, vec_reason = _maybe_build_vec_indexer(root, store)
    # Don't auto-bootstrap embeddings — costs $ and time. Use the admin
    # CLI: `amperstand-admin vec-rebuild`.

    return Indexers(store=store, fts=fts_indexer, vec=vec_indexer, vec_reason=vec_reason)


def _maybe_build_vec_indexer(
    root: Path, store: MarkdownStore,
) -> tuple[VectorIndexer | None, str | None]:
    """Construct the vector indexer if the prerequisites are present.

    - an embedding provider configured via AMPERSTAND_EMBED_* (default:
      OpenAI, which needs OPENAI_API_KEY)
    - sqlite-vec extension loadable (otherwise the index can't open)

    Either missing logs a one-line warning and returns (None, reason); the
    rest of the system continues to work with FTS-only search. An index
    built by a *different* model is opened anyway: queries against it
    answer 503 with the mismatch until `vec-rebuild --force` runs.
    """
    try:
        cfg = embed_config_from_env()
    except EmbeddingConfigError as exc:
        log.warning("embedding config invalid, vec search disabled: %s", exc)
        return None, str(exc)
    if not cfg.enabled:
        log.info("AMPERSTAND_EMBED_PROVIDER=none — semantic search disabled")
        return None, "AMPERSTAND_EMBED_PROVIDER=none"
    if cfg.provider == "openai" and not cfg.api_key:
        log.info("OPENAI_API_KEY not set — semantic search disabled")
        return None, "OPENAI_API_KEY not set"
    try:
        embedder = build_embedder(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not build embedder, vec search disabled: %s", exc)
        return None, str(exc)
    if embedder is None:  # pragma: no cover - provider=none handled above
        return None, "no embedding provider"
    try:
        vec_index = VectorIndex(root / ".store" / "vec.db", signature=embedder.signature)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open vec_index, vec search disabled: %s", exc)
        return None, str(exc)
    log.info("vector search: %s (index %s)", embedder.signature,
             vec_index.stored_signature() or "unstamped")
    return VectorIndexer(store=store, index=vec_index, embedder=embedder), None


@lru_cache(maxsize=1)
def get_reranker() -> BaseReranker:
    """The configured reranker (cached). A misconfiguration never takes the
    server down: it degrades to `NoReranker` carrying the reason, so a
    request that asks for rerank gets a 503 that says why."""
    try:
        reranker = build_reranker_from_env()
    except RerankError as exc:
        log.warning("rerank config invalid, rerank disabled: %s", exc)
        reranker = NoReranker()
        reranker.unavailable_reason = lambda exc=exc: f"rerank misconfigured: {exc}"  # type: ignore[method-assign]
        return reranker
    log.info("rerank: %s", reranker.describe())
    return reranker


def get_indexer() -> SearchIndexer:
    """Back-compat shim — many call sites still expect the FTS indexer."""
    return get_indexers().fts


def get_store() -> MarkdownStore:
    return get_indexers().store


def get_vec_indexer() -> VectorIndexer | None:
    return get_indexers().vec


def get_vec_unavailable_reason() -> str | None:
    return get_indexers().vec_reason


def reset_store_cache() -> None:
    """Clear the cached singletons. Tests use this between cases."""
    get_indexers.cache_clear()
    get_reranker.cache_clear()
