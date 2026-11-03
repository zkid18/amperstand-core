"""Rerankers: LLM grader (stubbed OpenAI client), HTTP cross-encoders in
every supported API shape (MockTransport), soft/hard failure, config."""

from __future__ import annotations

import json

import httpx
import pytest

from amperstand_core.search.models import SearchResult
from amperstand_core.search.rerank import (
    HTTPCrossEncoderReranker,
    LLMReranker,
    NoReranker,
    RerankError,
    build_reranker_from_env,
    rerank_config_from_env,
    rerank_with_llm,
)


def _cands(n=4):
    return [SearchResult(doc_id=f"d{i}", section_title=f"t{i}", section_path=["Doc", f"t{i}"],
                         snippet=f"snippet {i}", score=-1.0 / (i + 1)) for i in range(n)]


def _lookup(c: SearchResult) -> str:
    return f"body of {c.doc_id} " + "x" * 2000


def _transport(handler):
    seen: list[httpx.Request] = []

    def _h(request):
        seen.append(request)
        return handler(request, json.loads(request.content or b"{}"))

    return httpx.MockTransport(_h), seen


# ── HTTP cross-encoders ──────────────────────────────────────────────


def test_tei_shape_reorders_by_score() -> None:
    def handler(r, b):
        assert b["texts"] and "documents" not in b
        return httpx.Response(200, json=[{"index": 2, "score": 0.9}, {"index": 0, "score": 0.5},
                                         {"index": 1, "score": 0.1}, {"index": 3, "score": 0.0}])

    t, seen = _transport(handler)
    rr = HTTPCrossEncoderReranker(provider="tei", base_url="http://127.0.0.1:8081", transport=t)
    out = rr.rerank("q", _cands(), _lookup, limit=3)
    assert out.applied and out.kind == "cross_encoder"
    assert [r.doc_id for r in out.results] == ["d2", "d0", "d1"]
    assert out.results[0].score == -0.9  # negative = lower is better convention
    body = json.loads(seen[0].content)
    assert body["query"] == "q" and body["truncate"] is True
    assert len(body["texts"][0]) <= 600 + len("Doc > t0\n")  # passage cap honoured
    assert str(seen[0].url) == "http://127.0.0.1:8081/rerank"


def test_infinity_cohere_jina_shape() -> None:
    def handler(r, b):
        assert b["documents"] and b["top_n"] == 4
        return httpx.Response(200, json={"results": [{"index": 3, "relevance_score": 0.99},
                                                     {"index": 1, "relevance_score": 0.4}]})

    for provider, base in [("infinity", "http://127.0.0.1:7997"), ("cohere", None), ("jina", None)]:
        t, seen = _transport(handler)
        kw = {"api_key": "k", "model": "rerank-x"} if provider != "infinity" else {"model": "ce"}
        rr = HTTPCrossEncoderReranker(provider=provider, base_url=base, transport=t, **kw)
        out = rr.rerank("q", _cands(), _lookup, limit=4)
        assert out.applied
        # scored ones first by score, unscored keep fused order after them
        assert [r.doc_id for r in out.results] == ["d3", "d1", "d0", "d2"]
        assert json.loads(seen[0].content)["model"] == kw["model"]
    assert str(seen[0].url) == "https://api.jina.ai/v1/rerank"


def test_voyage_shape_uses_top_k_and_data() -> None:
    def handler(r, b):
        assert b["top_k"] == 4 and "top_n" not in b
        return httpx.Response(200, json={"data": [{"index": 1, "relevance_score": 0.8}]})

    t, seen = _transport(handler)
    rr = HTTPCrossEncoderReranker(provider="voyage", api_key="k", model="rerank-2.5", transport=t)
    out = rr.rerank("q", _cands(), _lookup, limit=2)
    assert [r.doc_id for r in out.results] == ["d1", "d0"]
    assert str(seen[0].url) == "https://api.voyageai.com/v1/rerank"
    assert seen[0].headers["authorization"] == "Bearer k"


def test_soft_failure_returns_fused_order_and_flags_it() -> None:
    t, _ = _transport(lambda r, b: httpx.Response(502, text="bad gateway"))
    rr = HTTPCrossEncoderReranker(provider="tei", base_url="http://h", transport=t)
    out = rr.rerank("q", _cands(), _lookup, limit=2)
    assert not out.applied
    assert [r.doc_id for r in out.results] == ["d0", "d1"]
    assert "502" in (out.error or "")


def test_hard_failure_raises() -> None:
    t, _ = _transport(lambda r, b: httpx.Response(500, text="boom"))
    rr = HTTPCrossEncoderReranker(provider="tei", base_url="http://h", transport=t, fail_mode="hard")
    with pytest.raises(RerankError, match="500"):
        rr.rerank("q", _cands(), _lookup, limit=2)


def test_empty_scores_is_a_soft_failure() -> None:
    t, _ = _transport(lambda r, b: httpx.Response(200, json={"results": []}))
    rr = HTTPCrossEncoderReranker(provider="infinity", base_url="http://h", transport=t)
    out = rr.rerank("q", _cands(), _lookup, limit=2)
    assert not out.applied and "no scores" in (out.error or "")


def test_vendor_rerankers_require_key_and_model() -> None:
    with pytest.raises(RerankError, match="API_KEY"):
        HTTPCrossEncoderReranker(provider="cohere", model="m")
    with pytest.raises(RerankError, match="MODEL"):
        HTTPCrossEncoderReranker(provider="jina", api_key="k")
    with pytest.raises(RerankError, match="BASE_URL"):
        HTTPCrossEncoderReranker(provider="tei")


# ── LLM grader ───────────────────────────────────────────────────────


class _StubOpenAI:
    last_kwargs: dict = {}
    script: list = []

    def __init__(self, **kwargs):
        _StubOpenAI.last_kwargs = kwargs
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Comp", (), {"create": staticmethod(self._create)})()

    @staticmethod
    def _create(**params):
        nxt = _StubOpenAI.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        msg = type("M", (), {"content": json.dumps(nxt)})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


@pytest.fixture
def stub_openai(monkeypatch):
    import openai
    monkeypatch.setattr(openai, "OpenAI", _StubOpenAI)
    _StubOpenAI.script = []
    return _StubOpenAI


def test_llm_reranker_scores_and_base_url(stub_openai) -> None:
    stub_openai.script = [{"scores": [{"id": 1, "score": 9}, {"id": 0, "score": 2}]}]
    rr = LLMReranker(api_key="sk", model="gpt-4o-mini", base_url="https://api.together.xyz/v1",
                     provider="openai_compatible")
    out = rr.rerank("q", _cands(3), _lookup, limit=3)
    assert out.applied and out.kind == "llm"
    assert [r.doc_id for r in out.results] == ["d1", "d0", "d2"]
    assert stub_openai.last_kwargs["base_url"] == "https://api.together.xyz/v1"


def test_llm_reranker_soft_fallback_on_error(stub_openai) -> None:
    stub_openai.script = [RuntimeError("429 insufficient_quota")]
    rr = LLMReranker(api_key="sk")
    out = rr.rerank("q", _cands(3), _lookup, limit=2)
    assert not out.applied and [r.doc_id for r in out.results] == ["d0", "d1"]
    assert "insufficient_quota" in (out.error or "")


def test_llm_reranker_without_key_is_unavailable() -> None:
    rr = LLMReranker(api_key=None)
    assert not rr.available
    assert "OPENAI_API_KEY" in (rr.unavailable_reason() or "")


def test_rerank_with_llm_backcompat(monkeypatch, stub_openai) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert [r.doc_id for r in rerank_with_llm("q", _cands(3), _lookup, limit=2)] == ["d0", "d1"]
    stub_openai.script = [{"scores": [{"id": 2, "score": 10}]}]
    assert [r.doc_id for r in rerank_with_llm("q", _cands(3), _lookup, limit=2, api_key="sk")] == ["d2", "d0"]


# ── config / factory ─────────────────────────────────────────────────


def test_rerank_config_defaults_to_llm_openai() -> None:
    cfg = rerank_config_from_env({"OPENAI_API_KEY": "sk"})
    assert (cfg.kind, cfg.provider, cfg.model, cfg.api_key) == ("llm", "openai", "gpt-4o-mini", "sk")
    assert cfg.fail_mode == "soft"


def test_rerank_config_validates_kind_provider_combos() -> None:
    with pytest.raises(RerankError):
        rerank_config_from_env({"AMPERSTAND_RERANK_KIND": "magic"})
    with pytest.raises(RerankError):
        rerank_config_from_env({"AMPERSTAND_RERANK_KIND": "llm", "AMPERSTAND_RERANK_PROVIDER": "tei"})
    with pytest.raises(RerankError):
        rerank_config_from_env({"AMPERSTAND_RERANK_KIND": "cross_encoder", "AMPERSTAND_RERANK_PROVIDER": "openai"})
    cfg = rerank_config_from_env({"AMPERSTAND_RERANK_KIND": "cross-encoder", "COHERE_API_KEY": "c",
                                  "AMPERSTAND_RERANK_PROVIDER": "cohere", "AMPERSTAND_RERANK_MODEL": "rerank-v3.5"})
    assert cfg.kind == "cross_encoder" and cfg.api_key == "c"


def test_factory_kinds(monkeypatch) -> None:
    monkeypatch.delenv("AMPERSTAND_RERANK_ENABLED", raising=False)
    assert isinstance(build_reranker_from_env({"AMPERSTAND_RERANK_KIND": "none"}), NoReranker)
    assert isinstance(build_reranker_from_env({"OPENAI_API_KEY": "sk"}), LLMReranker)
    ce = build_reranker_from_env({"AMPERSTAND_RERANK_KIND": "cross_encoder",
                                  "AMPERSTAND_RERANK_BASE_URL": "http://127.0.0.1:8081"})
    assert isinstance(ce, HTTPCrossEncoderReranker) and ce.provider == "tei"
    monkeypatch.setenv("AMPERSTAND_RERANK_ENABLED", "0")
    assert isinstance(build_reranker_from_env({"OPENAI_API_KEY": "sk"}), NoReranker)
