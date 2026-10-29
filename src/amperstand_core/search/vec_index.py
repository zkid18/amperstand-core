"""SQLite-backed vector index for semantic section search.

Uses the sqlite-vec extension's `vec0` virtual table for cosine-distance
KNN, plus a sidecar metadata table that maps each vector rowid back to
its (doc_id, section_path, section_title, content_hash). Lives in its
own SQLite file under .store/ so it can be rebuilt or torn down without
touching the FTS5 index next door.

Each row in the index represents one section of a doc. Reindexing a doc
deletes all rows whose doc_id matches and re-inserts new ones.

Model safety: vectors are only comparable within one embedding space, so
the index records the `EmbedSignature` (provider/model/dim) it was built
with in `vec_meta_kv`. Opening it with a different configured signature
does not destroy anything — reads and writes raise `IndexSignatureMismatch`
(→ HTTP 503) until the operator runs `amperstand-admin vec-rebuild --force`
or restores the matching configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Sequence

from amperstand_core.embeddings import EMBED_DIM, LEGACY_SIGNATURE, EmbedSignature
from amperstand_core.search.errors import SearchError
from amperstand_core.search.models import Section, SearchResult

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # bumped from 1: dim 1536 → 512 (Matryoshka)

META_PROVIDER = "embed_provider"
META_MODEL = "embed_model"
META_DIM = "embed_dim"


class IndexSignatureMismatch(SearchError):
    """The on-disk index was built by a different embedding model/dim."""

    def __init__(self, stored: EmbedSignature, configured: EmbedSignature) -> None:
        self.stored = stored
        self.configured = configured
        super().__init__(
            f"vector index was built with {stored} but the server is configured "
            f"for {configured}; run `amperstand-admin vec-rebuild --force` to "
            f"re-embed, or restore the matching AMPERSTAND_EMBED_* settings"
        )


def _sqlite_with_extensions():
    """Get a sqlite3 module that supports load_extension.

    macOS' python.org Python ships SQLite without extension loading
    enabled. Homebrew Python and Linux distro Python both ship it on.
    Surface a clear error message rather than a cryptic AttributeError.
    """
    import sqlite3
    if hasattr(sqlite3.Connection, "enable_load_extension"):
        return sqlite3
    raise SearchError(
        "Your sqlite3 module doesn't support load_extension — required for "
        "sqlite-vec. On macOS python.org Python this is disabled at build "
        "time; install Homebrew python (`brew install python@3.13`) or use "
        "Linux Python."
    )


def _vector_to_blob(vector: Sequence[float]) -> bytes:
    """sqlite-vec stores vectors as a tight float32 blob."""
    return struct.pack(f"{len(vector)}f", *vector)


def hash_section(section: Section) -> str:
    h = hashlib.sha256()
    h.update((section.title or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(section.body.encode("utf-8"))
    return f"sha256:{h.hexdigest()}"


def section_text_for_embedding(section: Section) -> str:
    """The string we hand to the embeddings API for a section.

    Prepend the section path so the embedding context includes the
    enclosing doc title + heading hierarchy. Helps disambiguation:
    "## Pricing" inside "Stripe vs Adyen" embeds differently from
    "## Pricing" inside "Why we picked Notion".
    """
    breadcrumb = " > ".join(section.path) if section.path else ""
    head = breadcrumb if breadcrumb else (section.title or "")
    body = section.body.strip()
    if head:
        return f"{head}\n\n{body}"
    return body


class VectorIndex:
    """sqlite-vec backed semantic index over heading-bounded sections.

    `signature` names the embedding space the caller will write/query
    with. Omit it (legacy callers) and the index assumes the pre-signature
    default, OpenAI text-embedding-3-small @512.
    """

    def __init__(
        self,
        path: Path,
        *,
        signature: EmbedSignature | None = None,
        dim: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if signature is None:
            signature = EmbedSignature(
                provider=LEGACY_SIGNATURE.provider,
                model=LEGACY_SIGNATURE.model,
                dim=dim or EMBED_DIM,
            )
        elif dim is not None and dim != signature.dim:
            raise SearchError(f"dim={dim} disagrees with signature {signature}")
        self.signature = signature
        self.dim = signature.dim
        self.mismatch: IndexSignatureMismatch | None = None

        sqlite3 = _sqlite_with_extensions()
        try:
            import sqlite_vec
        except ImportError as exc:
            raise SearchError(
                "sqlite-vec not installed. Run: pip install sqlite-vec"
            ) from exc

        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_meta_kv ("
                "  key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self._conn.execute(
                "SELECT value FROM vec_meta_kv WHERE key='schema_version'"
            ).fetchone()
            if row is not None and int(row[0]) != SCHEMA_VERSION:
                self._drop_tables_locked()
                row = None
            if row is None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO vec_meta_kv(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
            stored = self._stored_signature_locked()
            if self._count_locked() == 0 and stored != self.signature:
                # Empty index (fresh, legacy-but-empty, or stamped by a model
                # that never got to write anything): nothing to protect, so
                # adopt the configured signature and recreate the vec0 table
                # at the configured dim.
                self._drop_vec_tables_locked()
                self._stamp_locked(self.signature)
                stored = self.signature
            if stored is None:
                # Populated index predating signatures — production's
                # vec.db. It can only have come from the legacy model.
                stored = LEGACY_SIGNATURE
                if stored == self.signature:
                    self._stamp_locked(stored)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_sections USING vec0("
                f"  embedding float[{stored.dim}])"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_section_meta ("
                "  rowid INTEGER PRIMARY KEY,"
                "  doc_id TEXT NOT NULL,"
                "  section_title TEXT,"
                "  section_path TEXT,"
                "  content_hash TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_section_meta_doc_id "
                "ON vec_section_meta(doc_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vec_section_meta_hash "
                "ON vec_section_meta(content_hash)"
            )
        if stored != self.signature:
            self.mismatch = IndexSignatureMismatch(stored, self.signature)
            logger.error("%s", self.mismatch)
        else:
            self.mismatch = None

    # ── signature bookkeeping ───────────────────────────────────────

    def _stored_signature_locked(self) -> EmbedSignature | None:
        rows = dict(
            self._conn.execute(
                "SELECT key, value FROM vec_meta_kv WHERE key IN (?,?,?)",
                (META_PROVIDER, META_MODEL, META_DIM),
            ).fetchall()
        )
        if META_MODEL not in rows or META_DIM not in rows:
            return None
        return EmbedSignature(
            provider=rows.get(META_PROVIDER, "openai"),
            model=rows[META_MODEL],
            dim=int(rows[META_DIM]),
        )

    def _stamp_locked(self, sig: EmbedSignature) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO vec_meta_kv(key,value) VALUES(?,?)",
            [(META_PROVIDER, sig.provider), (META_MODEL, sig.model), (META_DIM, str(sig.dim))],
        )

    def _count_locked(self) -> int:
        has_meta = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_section_meta'"
        ).fetchone()
        if not has_meta:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM vec_section_meta").fetchone()[0]

    def _drop_vec_tables_locked(self) -> None:
        self._conn.execute("DROP TABLE IF EXISTS vec_section_meta")
        self._conn.execute("DROP TABLE IF EXISTS vec_sections")

    def _drop_tables_locked(self) -> None:
        self._drop_vec_tables_locked()
        self._conn.execute("DROP TABLE IF EXISTS vec_meta_kv")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_meta_kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def stored_signature(self) -> EmbedSignature | None:
        """What the on-disk index says it was built with (None = legacy, unstamped)."""
        return self._stored_signature_locked()

    def check_signature(self) -> None:
        """Raise `IndexSignatureMismatch` if this index can't serve the configured model."""
        if self.mismatch is not None:
            raise self.mismatch

    # ── writes ──────────────────────────────────────────────────────

    def existing_section_hashes(self, doc_id: str) -> set[str]:
        """Return the content_hashes already indexed for a given doc.

        Used by the indexer to skip re-embedding sections whose body
        didn't change between updates.
        """
        rows = self._conn.execute(
            "SELECT content_hash FROM vec_section_meta WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        return {r[0] for r in rows}

    def upsert_doc_sections(
        self,
        doc_id: str,
        sections: list[Section],
        embeddings: list[list[float]],
    ) -> None:
        """Replace all vectors for this doc atomically.

        `embeddings[i]` corresponds to `sections[i]`. Caller is responsible
        for skipping unchanged sections (see existing_section_hashes).
        """
        self.check_signature()
        if len(sections) != len(embeddings):
            raise SearchError(
                f"sections/embeddings length mismatch: "
                f"{len(sections)} vs {len(embeddings)}"
            )
        with self._conn:
            self._delete_doc_locked(doc_id)
            for section, vec in zip(sections, embeddings):
                if len(vec) != self.dim:
                    raise SearchError(
                        f"embedding has {len(vec)} dims, expected {self.dim}"
                    )
                cur = self._conn.execute(
                    "INSERT INTO vec_sections(embedding) VALUES (?)",
                    (_vector_to_blob(vec),),
                )
                self._conn.execute(
                    "INSERT INTO vec_section_meta"
                    "(rowid, doc_id, section_title, section_path, content_hash)"
                    " VALUES(?,?,?,?,?)",
                    (
                        cur.lastrowid,
                        doc_id,
                        section.title or "",
                        json.dumps(section.path, ensure_ascii=False),
                        hash_section(section),
                    ),
                )

    def delete_doc(self, doc_id: str) -> None:
        with self._conn:
            self._delete_doc_locked(doc_id)

    def _delete_doc_locked(self, doc_id: str) -> None:
        self._conn.execute(
            "DELETE FROM vec_sections WHERE rowid IN ("
            " SELECT rowid FROM vec_section_meta WHERE doc_id = ?)",
            (doc_id,),
        )
        self._conn.execute(
            "DELETE FROM vec_section_meta WHERE doc_id = ?", (doc_id,)
        )

    def reset(self) -> None:
        """Drop every vector and re-stamp the index with the configured
        signature. This is how a model switch is made: reset, then re-embed."""
        with self._conn:
            self._drop_tables_locked()
        self._init_schema()

    # ── reads ───────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return self._conn.execute(
            "SELECT COUNT(*) FROM vec_section_meta"
        ).fetchone()[0] == 0

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM vec_section_meta"
        ).fetchone()[0]

    def doc_centroid(self, doc_id: str) -> list[float] | None:
        """Mean of all section embeddings for a doc, L2-normalized.

        Used by the "find related" endpoint as a single-vector summary
        of the whole doc. Returns None if the doc has no sections in the
        index (e.g. failed to embed).
        """
        rows = self._conn.execute(
            "SELECT v.embedding FROM vec_sections v"
            " JOIN vec_section_meta m ON m.rowid = v.rowid"
            " WHERE m.doc_id = ?",
            (doc_id,),
        ).fetchall()
        if not rows:
            return None
        # Unpack each blob, sum, divide. Use the *stored* dim: related-docs
        # lookups work on stored vectors even when the configured model
        # differs (no query embedding is involved).
        n = len(rows[0][0]) // 4
        accum = [0.0] * n
        for (blob,) in rows:
            vec = struct.unpack(f"{n}f", blob)
            for i in range(n):
                accum[i] += vec[i]
        count = len(rows)
        mean = [x / count for x in accum]
        # L2-normalize so cosine-style MATCH behaves consistently.
        norm = sum(x * x for x in mean) ** 0.5
        if norm == 0:
            return mean
        return [x / norm for x in mean]

    def search(self, query_embedding: list[float], *, limit: int = 20) -> list[SearchResult]:
        if not 1 <= limit <= 1000:
            raise SearchError("limit out of range (1..1000)")
        stored_dim = self.mismatch.stored.dim if self.mismatch else self.dim
        if len(query_embedding) != stored_dim:
            raise SearchError(
                f"query embedding has {len(query_embedding)} dims, expected {stored_dim}"
            )
        rows = self._conn.execute(
            "SELECT v.rowid, m.doc_id, m.section_title, m.section_path, v.distance "
            " FROM vec_sections v "
            " JOIN vec_section_meta m ON m.rowid = v.rowid "
            " WHERE v.embedding MATCH ? AND k = ? "
            " ORDER BY v.distance",
            (_vector_to_blob(query_embedding), limit),
        ).fetchall()
        results: list[SearchResult] = []
        for _rowid, doc_id, title, path_json, distance in rows:
            try:
                path = json.loads(path_json) if path_json else []
            except json.JSONDecodeError:
                path = []
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    section_title=title or None,
                    section_path=path,
                    snippet="",  # no snippet from vec; caller can fetch from store
                    score=float(distance),
                )
            )
        return results

    def search_query(self, query_embedding: list[float], *, limit: int = 20) -> list[SearchResult]:
        """`search()` for a *freshly embedded query*: refuses on a model
        mismatch, because the query vector lives in the wrong space."""
        self.check_signature()
        return self.search(query_embedding, limit=limit)

    def close(self) -> None:
        self._conn.close()
