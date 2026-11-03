"""Pydantic request/response models for the vault API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateDocRequest(BaseModel):
    body: str
    frontmatter: dict[str, Any] | None = None


class UpdateDocRequest(BaseModel):
    body: str
    frontmatter: dict[str, Any] | None = None


class DocMetaResponse(BaseModel):
    """Identity + metadata of a doc.

    Schema unification note: PKM v2 friction inventory flagged drift between
    the on-disk frontmatter (which uses Obsidian-friendly short names
    `type`/`captured`/`updated` — Dataview reads from these) and the JSON
    response (which historically used Pydantic-style `content_type`/
    `captured_at`/`updated_at`). Both name sets are now emitted with
    identical values so callers can pick whichever fits. The on-disk names
    are the canonical / preferred ones for new code; the *_at variants are
    retained for backward compat and will likely stay forever — they're
    cheap.
    """

    id: str
    path: str
    title: str | None = None
    source: str | None = None
    # Canonical (matches on-disk frontmatter / Obsidian's Dataview style).
    type: str | None = None
    captured: datetime
    updated: datetime
    # Legacy snake-case-at names. Same values as the canonical fields above.
    content_type: str | None = None
    captured_at: datetime
    updated_at: datetime
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    content_hash: str
    body_hash: str | None = None


class DocResponse(DocMetaResponse):
    body: str


class AssetResponse(BaseModel):
    """Returned after attaching an asset to a doc."""

    doc_id: str
    link: str  # markdown-relative link to embed in the doc body


class ListResponse(BaseModel):
    items: list[DocMetaResponse]
    next_cursor: str | None = None


class SearchRequest(BaseModel):
    q: str
    limit: int = Field(default=20, ge=1, le=100)
    # `any`        — OR-joins every word, BM25 ranks. Friendliest UX.
    # `fts`        — power-user, passes the query verbatim (AND default,
    #                supports `foo OR bar`, `"phrase"`, `prefix*`).
    # `substring`  — every token treated as a literal phrase (no operators).
    # `semantic`   — embed the query, KNN over section vectors. Catches
    #                conceptual matches keywords miss; needs OPENAI_API_KEY
    #                + a populated vec_index (run `amperstand-admin vec-rebuild`).
    # `hybrid`     — combine FTS BM25 and semantic via reciprocal rank
    #                fusion. Best general-purpose mode when both indexes
    #                are available.
    mode: str = Field(default="any", pattern=r"^(any|fts|substring|semantic|hybrid)$")


class SearchHit(BaseModel):
    doc_id: str
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    snippet: str
    score: float


class SearchMeta(BaseModel):
    """How a search was actually answered — so a caller can tell a real
    hybrid result from a BM25 fallback or an un-reranked list."""

    mode: str
    embed_provider: str | None = None
    embed_model: str | None = None
    rerank_requested: bool = False
    rerank_applied: bool = False
    rerank_kind: str | None = None
    # "bm25" when vectors were unavailable and the caller opted in with
    # allow_bm25_fallback — never set silently.
    fallback: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchHit] = Field(default_factory=list)
    meta: SearchMeta | None = None


class SemanticSearchRequest(BaseModel):
    """Dedicated body for POST /vault/search/semantic — same as a SearchRequest
    with mode pinned to semantic (no mode field, fewer ways to misuse)."""

    q: str
    limit: int = Field(default=20, ge=1, le=100)


class HybridSearchRequest(BaseModel):
    """Dedicated body for POST /vault/search/hybrid."""

    q: str
    limit: int = Field(default=20, ge=1, le=100)
    rerank: bool = Field(
        default=False,
        description=(
            "Adds an LLM re-rank stage after RRF fusion. Expands the candidate "
            "pool, asks gpt-4o-mini to score relevance, returns top `limit` by "
            "LLM judgement. +1-2s latency, ~$0.001/query. Server-wide kill "
            "switch: AMPERSTAND_RERANK_ENABLED=0."
        ),
    )
    allow_bm25_fallback: bool = Field(
        default=False,
        description=(
            "Explicit opt-in: when the embedding provider is down (or no "
            "vector index is configured) answer with BM25-only results and "
            "meta.fallback='bm25' instead of a 503. Off by default so a "
            "hybrid request never silently degrades."
        ),
    )
