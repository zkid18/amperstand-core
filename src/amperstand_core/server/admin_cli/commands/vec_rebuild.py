"""amperstand-admin vec-rebuild — backfill the semantic search index.

Walks every doc in the vault, parses sections, embeds them via the
configured embeddings provider (AMPERSTAND_EMBED_*; OpenAI by default),
and inserts vectors into .store/vec.db. Skips docs whose section
content_hashes already match (incremental).

The index records which provider/model/dim it was built with. If that
differs from the configured provider the command refuses to mix vector
spaces and asks for --force, which resets the index first and re-embeds
everything under the new signature.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from amperstand_core.embeddings import (
    EmbeddingConfigError,
    build_embedder,
    embed_config_from_env,
)
from amperstand_core.search import VectorIndex, VectorIndexer
from amperstand_core.search.parser import parse_sections
from amperstand_core.search.vec_index import (
    hash_section,
    section_text_for_embedding,
)
from amperstand_core.store import MarkdownStore

from amperstand_core.server.admin_cli.config import AdminConfig

log = logging.getLogger(__name__)


def run(config: AdminConfig, *, force: bool = False, dry_run: bool = False) -> None:
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    env = dict(config.env)
    if config.openai_api_key and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = config.openai_api_key
    try:
        cfg = embed_config_from_env(env)
    except EmbeddingConfigError as exc:
        raise SystemExit(f"embedding config error: {exc}")
    if not cfg.enabled:
        raise SystemExit("AMPERSTAND_EMBED_PROVIDER=none — nothing to embed.")
    if cfg.provider == "openai" and not cfg.api_key:
        raise SystemExit("OPENAI_API_KEY not set in env-file. Cannot embed.")

    try:
        embedder = build_embedder(cfg)
    except EmbeddingConfigError as exc:
        raise SystemExit(f"embedding provider not usable: {exc}")
    assert embedder is not None
    sig = embedder.signature

    print(
        f"Vector rebuild starting (provider={sig.provider} model={sig.model} "
        f"dim={sig.dim}, force={force}, dry_run={dry_run})"
    )
    print(f"Data dir: {data_dir}")

    # Build a store with no on_change hook — we don't want this run to re-fire
    # the live indexers (they'd duplicate work). We're populating the vec index
    # directly via the indexer.
    store = MarkdownStore(data_dir, on_change=None)
    vec_index = VectorIndex(data_dir / ".store" / "vec.db", signature=sig)
    stored = vec_index.stored_signature()
    print(f"Index signature on disk: {stored or 'unstamped (legacy)'} — {vec_index.count()} vectors")

    if vec_index.mismatch is not None and not force:
        raise SystemExit(
            f"refusing to mix vector spaces: {vec_index.mismatch}\n"
            f"Re-run with --force to drop the index and re-embed with {sig}."
        )

    if dry_run:
        # Just count work to be done.
        total = sum(1 for _ in store.iter_all())
        action = "would reset the index and re-embed" if (force or vec_index.mismatch) else "would scan"
        print(f"{action} {total} docs with {sig}")
        return

    indexer = VectorIndexer(store=store, index=vec_index, embedder=embedder)
    _ = indexer  # the loop below mirrors indexer._reindex_doc but with progress output

    if force:
        print("--force: dropping existing vec index")
        vec_index.reset()

    indexed = 0
    skipped_unchanged = 0
    failed = 0
    started = time.time()
    embedded_sections = 0

    for meta in store.iter_all():
        try:
            doc = store.get(meta.id)
        except Exception:
            failed += 1
            log.exception("could not load doc %s", meta.id)
            continue

        sections = parse_sections(doc.body, doc.meta.title)
        if not sections:
            vec_index.delete_doc(meta.id)
            continue

        new_hashes = {hash_section(s) for s in sections}
        existing = vec_index.existing_section_hashes(meta.id)
        if new_hashes == existing and existing:
            skipped_unchanged += 1
            continue

        try:
            texts = [section_text_for_embedding(s) for s in sections]
            embeddings = embedder.embed_batch(texts)
            vec_index.upsert_doc_sections(meta.id, sections, embeddings)
            indexed += 1
            embedded_sections += len(sections)
        except Exception:
            failed += 1
            log.exception("embed failed for doc %s", meta.id)
            continue

        if indexed % 50 == 0 and indexed > 0:
            elapsed = time.time() - started
            rate = indexed / elapsed if elapsed else 0
            print(
                f"  indexed={indexed} skipped_unchanged={skipped_unchanged} "
                f"failed={failed} sections={embedded_sections} "
                f"rate={rate:.1f} docs/s"
            )

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.0f}s. indexed={indexed} skipped_unchanged={skipped_unchanged} "
        f"failed={failed} sections={embedded_sections}"
    )
    print(f"vec index now holds {vec_index.count()} section vectors ({sig})")
