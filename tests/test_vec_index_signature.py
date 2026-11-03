"""VectorIndex records the embedding signature and refuses to mix spaces.
Needs sqlite-vec loadable in this interpreter (skipped otherwise)."""

from __future__ import annotations

import pytest

from amperstand_core.embeddings import LEGACY_SIGNATURE, EmbedSignature
from amperstand_core.search import IndexSignatureMismatch, VectorIndex
from amperstand_core.search.models import Section

from tests.fakes import sqlite_vec_loadable

pytestmark = pytest.mark.skipif(not sqlite_vec_loadable(), reason="sqlite-vec not loadable here")

SIG_A = EmbedSignature("tei", "granite-97m", 4)
SIG_B = EmbedSignature("openai", "text-embedding-3-small", 4)


def _sec(i: int) -> Section:
    return Section(title=f"s{i}", level=1, path=["Doc", f"s{i}"], body=f"body {i}")


def _vec(i: int, dim: int = 4) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def test_fresh_index_adopts_configured_signature(tmp_path) -> None:
    idx = VectorIndex(tmp_path / "vec.db", signature=SIG_A)
    assert idx.stored_signature() == SIG_A
    assert idx.mismatch is None
    idx.upsert_doc_sections("d1", [_sec(0), _sec(1)], [_vec(0), _vec(1)])
    assert idx.count() == 2
    assert idx.search_query(_vec(0), limit=1)[0].doc_id == "d1"
    idx.close()


def test_reopen_with_other_model_is_a_mismatch_not_a_reset(tmp_path) -> None:
    idx = VectorIndex(tmp_path / "vec.db", signature=SIG_A)
    idx.upsert_doc_sections("d1", [_sec(0)], [_vec(0)])
    idx.close()

    other = VectorIndex(tmp_path / "vec.db", signature=SIG_B)
    assert other.count() == 1, "vectors must survive a config change"
    assert isinstance(other.mismatch, IndexSignatureMismatch)
    assert other.mismatch.stored == SIG_A and other.mismatch.configured == SIG_B
    with pytest.raises(IndexSignatureMismatch, match="vec-rebuild --force"):
        other.search_query(_vec(0))
    with pytest.raises(IndexSignatureMismatch):
        other.upsert_doc_sections("d2", [_sec(1)], [_vec(1)])
    # Stored-vector reads (related docs) still work: no query vector involved.
    assert other.doc_centroid("d1") is not None
    assert other.search(_vec(0), limit=1)[0].doc_id == "d1"
    other.close()


def test_reset_switches_signature(tmp_path) -> None:
    idx = VectorIndex(tmp_path / "vec.db", signature=SIG_A)
    idx.upsert_doc_sections("d1", [_sec(0)], [_vec(0)])
    idx.close()
    other = VectorIndex(tmp_path / "vec.db", signature=SIG_B)
    assert other.mismatch is not None
    other.reset()
    assert other.mismatch is None and other.count() == 0
    assert other.stored_signature() == SIG_B
    other.upsert_doc_sections("d1", [_sec(0)], [_vec(0)])
    assert other.search_query(_vec(0), limit=1)[0].doc_id == "d1"
    other.close()


def test_legacy_unstamped_index_is_assumed_openai_512(tmp_path) -> None:
    legacy_cfg = EmbedSignature(LEGACY_SIGNATURE.provider, LEGACY_SIGNATURE.model, 512)
    idx = VectorIndex(tmp_path / "vec.db", signature=legacy_cfg)
    idx.upsert_doc_sections("d1", [_sec(0)], [_vec(0, 512)])
    # Simulate production's pre-signature vec.db: schema_version only.
    with idx._conn:
        idx._conn.execute("DELETE FROM vec_meta_kv WHERE key != 'schema_version'")
    idx.close()

    # Default (legacy) configuration: adopted and stamped, no mismatch.
    same = VectorIndex(tmp_path / "vec.db")  # no signature → legacy default
    assert same.mismatch is None and same.count() == 1
    assert same.stored_signature() == legacy_cfg
    same.close()


def test_legacy_unstamped_index_vs_new_model_is_a_mismatch(tmp_path) -> None:
    legacy_cfg = EmbedSignature(LEGACY_SIGNATURE.provider, LEGACY_SIGNATURE.model, 512)
    idx = VectorIndex(tmp_path / "vec.db", signature=legacy_cfg)
    idx.upsert_doc_sections("d1", [_sec(0)], [_vec(0, 512)])
    with idx._conn:
        idx._conn.execute("DELETE FROM vec_meta_kv WHERE key != 'schema_version'")
    idx.close()
    new = VectorIndex(tmp_path / "vec.db", signature=EmbedSignature("tei", "granite-97m", 384))
    assert new.mismatch is not None
    assert new.mismatch.stored == legacy_cfg
    assert new.stored_signature() is None  # never stamped with a wrong guess
    new.close()


def test_empty_index_with_other_dim_is_recreated(tmp_path) -> None:
    idx = VectorIndex(tmp_path / "vec.db", signature=EmbedSignature("tei", "m", 4))
    idx.close()
    idx2 = VectorIndex(tmp_path / "vec.db", signature=EmbedSignature("tei", "m", 2))
    assert idx2.mismatch is None
    idx2.upsert_doc_sections("d1", [_sec(0)], [[1.0, 0.0]])
    assert idx2.search_query([1.0, 0.0], limit=1)[0].doc_id == "d1"
    idx2.close()


def test_dim_argument_must_agree_with_signature(tmp_path) -> None:
    from amperstand_core.search import SearchError
    with pytest.raises(SearchError):
        VectorIndex(tmp_path / "vec.db", signature=SIG_A, dim=8)
