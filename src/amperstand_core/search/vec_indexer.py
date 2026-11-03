"""Subscribe to MarkdownStore change events; keep VectorIndex in sync.

Mirrors the design of SearchIndexer (FTS5) but for embeddings:
- on CREATED/UPDATED: parse sections, embed any whose body changed,
  reuse existing vectors for unchanged sections.
- on DELETED: drop all vectors for the doc.

Splitting the FTS and Vector indexers — rather than one combined hook —
lets a failed embedding (rate-limit, network blip) leave FTS healthy.
"""

from __future__ import annotations

import logging

from amperstand_core.embeddings import BaseEmbedder, EmbeddingError
from amperstand_core.search.parser import parse_sections
from amperstand_core.search.vec_index import (
    VectorIndex,
    hash_section,
    section_text_for_embedding,
)
from amperstand_core.store.events import ChangeEvent, ChangeKind
from amperstand_core.store.store import MarkdownStore

logger = logging.getLogger(__name__)


class VectorIndexer:
    def __init__(
        self,
        store: MarkdownStore,
        index: VectorIndex,
        embedder: BaseEmbedder,
    ) -> None:
        self._store = store
        self._index = index
        self._embedder = embedder

    @property
    def store(self) -> MarkdownStore:
        return self._store

    @property
    def index(self) -> VectorIndex:
        return self._index

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    def handle_change(self, event: ChangeEvent) -> None:
        try:
            if event.kind in (ChangeKind.CREATED, ChangeKind.UPDATED):
                self._reindex_doc(event.id)
            elif event.kind == ChangeKind.DELETED:
                self._index.delete_doc(event.id)
        except EmbeddingError:
            logger.exception("vec indexer: embedding failed for %s", event.id)
        except Exception:
            logger.exception("vec indexer: failed for %s (%s)", event.id, event.kind)

    def bootstrap(self, *, force: bool = False) -> int:
        """Embed every doc the store knows about. Returns count.

        force=False: no-op if the index already has rows.
        force=True : reset the index then rebuild from scratch.
        """
        if not force and not self._index.is_empty():
            return 0
        if force:
            self._index.reset()
        count = 0
        for meta in self._store.iter_all():
            try:
                self._reindex_doc(meta.id)
                count += 1
            except Exception:
                logger.exception("vec bootstrap failed for %s", meta.id)
        return count

    def _reindex_doc(self, doc_id: str) -> None:
        doc = self._store.get(doc_id)
        sections = parse_sections(doc.body, doc.meta.title)
        if not sections:
            self._index.delete_doc(doc_id)
            return

        # Skip the embedding round-trip when every section is byte-identical
        # to what's already in the index.
        new_hashes = {hash_section(s) for s in sections}
        existing_hashes = self._index.existing_section_hashes(doc_id)
        if new_hashes == existing_hashes and existing_hashes:
            return

        texts = [section_text_for_embedding(s) for s in sections]
        embeddings = self._embedder.embed_batch(texts)
        self._index.upsert_doc_sections(doc_id, sections, embeddings)
