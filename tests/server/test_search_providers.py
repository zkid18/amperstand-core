"""Search endpoints with pluggable providers: explicit 503s, opt-in BM25
fallback, index signature guard, reranker kinds. Uses FakeEmbedder (no
network) injected through the store factory, and MockTransport rerankers."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from amperstand_core.embeddings import EmbedSignature
from amperstand_core.search import VectorIndex
from amperstand_core.search.rerank import HTTPCrossEncoderReranker
from amperstand_core.server.app import app, reset_job_store_cache
from amperstand_core.server.vault_api import store_factory

from tests.fakes import FakeEmbedder, sqlite_vec_loadable

HEAD = {"Authorization": "Bearer devkey"}
needs_vec = pytest.mark.skipif(not sqlite_vec_loadable(), reason="sqlite-vec not loadable here")


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """Route the store factory's provider construction to a FakeEmbedder.
    The env still has to *look* configured (provider + key) so the factory
    gets as far as building one."""
    emb = FakeEmbedder(dim=8)
    monkeypatch.setenv("AMPERSTAND_EMBED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AMPERSTAND_EMBED_BASE_URL", "http://fake.invalid")
    monkeypatch.setenv("AMPERSTAND_EMBED_MODEL", "fake-model")
    monkeypatch.setenv("AMPERSTAND_EMBED_DIM", "8")
    monkeypatch.setattr(store_factory, "build_embedder", lambda cfg, **kw: emb)
    return emb


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AMPERSTAND_API_KEY", "devkey")
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AMPERSTAND_RERANK_KIND", raising=False)
    monkeypatch.delenv("AMPERSTAND_RERANK_ENABLED", raising=False)
    store_factory.reset_store_cache()
    reset_job_store_cache()
    with TestClient(app) as c:
        yield c
    store_factory.reset_store_cache()
    reset_job_store_cache()


def _seed(client: TestClient) -> None:
    client.post("/vault", headers=HEAD, json={"body": "# Apples\n\nred apple orchard harvest\n",
                                              "frontmatter": {"title": "Apples"}})
    client.post("/vault", headers=HEAD, json={"body": "# Trains\n\nsteam locomotive railway timetable\n",
                                              "frontmatter": {"title": "Trains"}})


# ── no provider configured (works without sqlite-vec) ────────────────


def test_hybrid_503_names_the_missing_key_by_default(client: TestClient) -> None:
    _seed(client)
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple", "limit": 5})
    assert r.status_code == 503
    d = r.json()["detail"]
    assert d["error"] == "feature_requires_openai_key"
    assert "OPENAI_API_KEY" in d["detail"]


def test_hybrid_bm25_fallback_is_opt_in_and_flagged(client: TestClient) -> None:
    _seed(client)
    r = client.post("/vault/search/hybrid", headers=HEAD,
                    json={"q": "apple", "limit": 5, "allow_bm25_fallback": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["fallback"] == "bm25"
    assert body["meta"]["mode"] == "hybrid"
    assert body["meta"]["warnings"] and "OPENAI_API_KEY" in body["meta"]["warnings"][0]
    assert body["results"] and body["results"][0]["doc_id"]


def test_legacy_search_hybrid_flags_its_fallback(client: TestClient) -> None:
    _seed(client)
    r = client.post("/vault/search", headers=HEAD, json={"q": "apple", "mode": "hybrid"})
    assert r.status_code == 200
    assert r.json()["meta"]["fallback"] == "bm25"
    r2 = client.post("/vault/search", headers=HEAD, json={"q": "apple", "mode": "any"})
    assert r2.json()["meta"] == {"mode": "any", "embed_provider": None, "embed_model": None,
                                 "rerank_requested": False, "rerank_applied": False,
                                 "rerank_kind": None, "fallback": None, "warnings": []}


def test_provider_none_gives_generic_unavailable(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("AMPERSTAND_EMBED_PROVIDER", "none")
    store_factory.reset_store_cache()
    r = client.post("/vault/search/semantic", headers=HEAD, json={"q": "apple"})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "vector_index_unavailable"
    assert "AMPERSTAND_EMBED_PROVIDER=none" in r.json()["detail"]["reason"]


def test_rerank_requested_with_kind_none_is_503(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("AMPERSTAND_RERANK_KIND", "none")
    store_factory.reset_store_cache()
    r = client.post("/vault/search/hybrid", headers=HEAD,
                    json={"q": "apple", "rerank": True, "allow_bm25_fallback": True})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "rerank_unavailable"


# ── with a (fake) provider and a real sqlite-vec index ───────────────


@needs_vec
def test_semantic_and_hybrid_work_with_a_pluggable_provider(client, fake_embedder) -> None:
    _seed(client)
    assert fake_embedder.calls, "indexer should embed new docs on create"
    assert fake_embedder.calls[0][0] == "document"
    r = client.post("/vault/search/semantic", headers=HEAD, json={"q": "apple orchard", "limit": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"] == {"mode": "semantic", "embed_provider": "fake", "embed_model": "fake-model",
                            "rerank_requested": False, "rerank_applied": False, "rerank_kind": None,
                            "fallback": None, "warnings": []}
    assert body["results"][0]["section_title"] == "Apples"
    assert fake_embedder.calls[-1][0] == "query"

    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "railway timetable", "limit": 2})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["section_title"] == "Trains"
    assert r.json()["meta"]["fallback"] is None


@needs_vec
def test_provider_failure_is_a_503_with_the_cause(client, fake_embedder) -> None:
    _seed(client)
    fake_embedder.fail_with = "HTTP 429: insufficient_quota"
    for path in ("/vault/search/semantic", "/vault/search/hybrid"):
        r = client.post(path, headers=HEAD, json={"q": "apple"})
        assert r.status_code == 503, r.text
        d = r.json()["detail"]
        assert d["error"] == "embedding_provider_unavailable"
        assert d["provider"] == "fake" and d["model"] == "fake-model"
        assert "insufficient_quota" in d["detail"]
    r = client.post("/vault/search", headers=HEAD, json={"q": "apple", "mode": "semantic"})
    assert r.status_code == 503 and r.json()["detail"]["error"] == "embedding_provider_unavailable"


@needs_vec
def test_provider_failure_with_opt_in_fallback(client, fake_embedder) -> None:
    _seed(client)
    fake_embedder.fail_with = "connection refused"
    r = client.post("/vault/search/hybrid", headers=HEAD,
                    json={"q": "apple", "limit": 3, "allow_bm25_fallback": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["fallback"] == "bm25"
    assert body["meta"]["embed_provider"] == "fake"
    assert any("connection refused" in w for w in body["meta"]["warnings"])
    assert body["results"][0]["section_title"] == "Apples"


@needs_vec
def test_index_built_by_another_model_is_refused(client, fake_embedder, tmp_path, monkeypatch) -> None:
    _seed(client)
    # Operator switches models without rebuilding: restart with a new signature.
    monkeypatch.setenv("AMPERSTAND_EMBED_MODEL", "other-model")
    fake_embedder.model = "other-model"
    store_factory.reset_store_cache()
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple"})
    assert r.status_code == 503, r.text
    d = r.json()["detail"]
    assert d["error"] == "vec_index_model_mismatch"
    assert d["index"] == {"provider": "fake", "model": "fake-model", "dim": 8}
    assert d["configured"] == {"provider": "fake", "model": "other-model", "dim": 8}
    assert "vec-rebuild --force" in d["detail"]
    # Opt-in fallback still answers.
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple", "allow_bm25_fallback": True})
    assert r.status_code == 200 and r.json()["meta"]["fallback"] == "bm25"
    # Nothing was destroyed: 2 docs x (preamble + heading section) = 4 vectors.
    idx = VectorIndex(tmp_path / ".store" / "vec.db", signature=EmbedSignature("fake", "fake-model", 8))
    assert idx.count() == 4 and idx.mismatch is None
    idx.close()


@needs_vec
def test_cross_encoder_rerank_applied_and_reported(client, fake_embedder, monkeypatch) -> None:
    _seed(client)
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        # Prefer the Trains passage whatever the query says.
        scores = [{"index": i, "score": (1.0 if "locomotive" in t else 0.1)} for i, t in enumerate(body["texts"])]
        return httpx.Response(200, json=scores)

    rr = HTTPCrossEncoderReranker(provider="tei", base_url="http://ce.invalid",
                                  transport=httpx.MockTransport(handler))
    monkeypatch.setattr(store_factory, "build_reranker_from_env", lambda: rr)
    store_factory.reset_store_cache()
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple", "limit": 2, "rerank": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["meta"]["rerank_requested"] and body["meta"]["rerank_applied"]
    assert body["meta"]["rerank_kind"] == "cross_encoder"
    assert body["results"][0]["section_title"] == "Trains"
    assert seen and seen[0]["query"] == "apple"


@needs_vec
def test_cross_encoder_soft_failure_is_flagged_hard_is_503(client, fake_embedder, monkeypatch) -> None:
    _seed(client)
    down = httpx.MockTransport(lambda req: httpx.Response(503, text="sidecar down"))
    soft = HTTPCrossEncoderReranker(provider="tei", base_url="http://ce.invalid", transport=down)
    monkeypatch.setattr(store_factory, "build_reranker_from_env", lambda: soft)
    store_factory.reset_store_cache()
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple", "limit": 2, "rerank": True})
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["rerank_applied"] is False
    assert any("sidecar down" in w for w in r.json()["meta"]["warnings"])

    hard = HTTPCrossEncoderReranker(provider="tei", base_url="http://ce.invalid", transport=down,
                                    fail_mode="hard")
    monkeypatch.setattr(store_factory, "build_reranker_from_env", lambda: hard)
    store_factory.reset_store_cache()
    r = client.post("/vault/search/hybrid", headers=HEAD, json={"q": "apple", "limit": 2, "rerank": True})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "rerank_provider_unavailable"


@needs_vec
def test_related_docs_work_even_when_model_mismatched(client, fake_embedder, monkeypatch) -> None:
    _seed(client)
    doc_id = client.post("/vault/search", headers=HEAD, json={"q": "apple", "mode": "any"}).json()["results"][0]["doc_id"]
    monkeypatch.setenv("AMPERSTAND_EMBED_MODEL", "other-model")
    fake_embedder.model = "other-model"
    store_factory.reset_store_cache()
    r = client.get(f"/vault/{doc_id}/related", headers=HEAD)
    assert r.status_code == 200, r.text
