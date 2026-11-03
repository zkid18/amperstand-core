"""Embedding providers: config, factory, request/response shapes, failure
classification, dim guard. All HTTP goes through httpx.MockTransport —
no network, no keys."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from amperstand_core.embeddings import (
    EMBED_DIM,
    Embedder,
    EmbeddingConfigError,
    EmbeddingError,
    EmbedSignature,
    OpenAIEmbedder,
    build_embedder,
    build_embedder_from_env,
    classify_failure,
    embed_config_from_env,
)
from amperstand_core.embeddings import http_provider as hp
from amperstand_core.embeddings.base import truncate_text


# ── config ───────────────────────────────────────────────────────────


def test_default_config_is_the_legacy_openai_setup() -> None:
    cfg = embed_config_from_env({"OPENAI_API_KEY": "sk-test"})
    assert cfg.provider == "openai"
    assert cfg.model == "text-embedding-3-small"
    assert cfg.dim == EMBED_DIM == 512
    assert cfg.request_dim is True
    assert cfg.api_key == "sk-test"
    assert cfg.matryoshka is False


def test_config_reads_provider_specific_env() -> None:
    cfg = embed_config_from_env({
        "AMPERSTAND_EMBED_PROVIDER": "tei",
        "AMPERSTAND_EMBED_BASE_URL": "http://127.0.0.1:8080/",
        "AMPERSTAND_EMBED_MODEL": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "AMPERSTAND_EMBED_DIM": "384",
        "AMPERSTAND_EMBED_QUERY_PREFIX": "query: ",
        "AMPERSTAND_EMBED_MAX_TOKENS": "2000",
    })
    assert cfg.provider == "tei"
    assert cfg.dim == 384
    assert cfg.request_dim is False  # auto → off for non-openai
    assert cfg.query_prefix == "query: "
    assert cfg.max_tokens == 2000


def test_config_key_fallbacks_per_vendor() -> None:
    assert embed_config_from_env({"AMPERSTAND_EMBED_PROVIDER": "voyage", "VOYAGE_API_KEY": "v"}).api_key == "v"
    assert embed_config_from_env({"AMPERSTAND_EMBED_PROVIDER": "jina", "JINA_API_KEY": "j"}).api_key == "j"
    assert embed_config_from_env({"AMPERSTAND_EMBED_PROVIDER": "cohere", "COHERE_API_KEY": "c"}).api_key == "c"
    # explicit key wins over the vendor fallback
    cfg = embed_config_from_env({"AMPERSTAND_EMBED_PROVIDER": "cohere", "COHERE_API_KEY": "c",
                                 "AMPERSTAND_EMBED_API_KEY": "explicit"})
    assert cfg.api_key == "explicit"


def test_config_rejects_unknown_provider_and_bad_dim() -> None:
    with pytest.raises(EmbeddingConfigError):
        embed_config_from_env({"AMPERSTAND_EMBED_PROVIDER": "bedrock"})
    with pytest.raises(EmbeddingConfigError):
        embed_config_from_env({"AMPERSTAND_EMBED_DIM": "many"})


def test_factory_requires_what_each_provider_needs() -> None:
    assert build_embedder_from_env({"AMPERSTAND_EMBED_PROVIDER": "none"}) is None
    with pytest.raises(EmbeddingConfigError, match="OPENAI_API_KEY"):
        build_embedder_from_env({})
    with pytest.raises(EmbeddingConfigError, match="BASE_URL"):
        build_embedder_from_env({"AMPERSTAND_EMBED_PROVIDER": "tei"})
    with pytest.raises(EmbeddingConfigError, match="MODEL"):
        build_embedder_from_env({"AMPERSTAND_EMBED_PROVIDER": "openai_compatible",
                                 "AMPERSTAND_EMBED_BASE_URL": "http://x"})
    with pytest.raises(EmbeddingConfigError, match="API_KEY"):
        build_embedder_from_env({"AMPERSTAND_EMBED_PROVIDER": "voyage",
                                 "AMPERSTAND_EMBED_MODEL": "voyage-3.5-lite"})
    with pytest.raises(EmbeddingConfigError, match="ONNX_DIR"):
        build_embedder_from_env({"AMPERSTAND_EMBED_PROVIDER": "onnx"})


# ── back-compat surface ──────────────────────────────────────────────


def test_embedder_alias_and_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Embedder is OpenAIEmbedder
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        Embedder.from_env()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    e = Embedder.from_env()
    assert e.model == "text-embedding-3-small"
    assert e.dim == 512
    assert e.signature == EmbedSignature("openai", "text-embedding-3-small", 512)
    assert "api_key" not in json.dumps(e.describe())


def test_truncate_text_respects_budget() -> None:
    long = "word " * 20000
    assert len(truncate_text(long, 100)) < len(long)
    assert truncate_text("short", 100) == "short"


# ── OpenAI SDK path (stubbed client) ─────────────────────────────────


class _Resp:
    def __init__(self, vectors):
        self.data = [type("D", (), {"index": i, "embedding": v})() for i, v in enumerate(vectors)]


class _StubEmbeddings:
    def __init__(self, script):
        self.script = list(script)  # each item: list-of-vectors or Exception
        self.calls: list[dict] = []

    def create(self, **params):
        self.calls.append(params)
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(nxt)


class _APIError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status_code = status


def _openai(monkeypatch, script, **kw):
    monkeypatch.setattr("time.sleep", lambda s: None)
    e = OpenAIEmbedder(api_key="sk-test", **kw)
    stub = _StubEmbeddings(script)
    e._client = type("C", (), {"embeddings": stub})()
    return e, stub


def test_openai_sends_model_and_dimensions_like_before(monkeypatch) -> None:
    e, stub = _openai(monkeypatch, [[[0.1] * 512]])
    assert e.embed("hello") == [0.1] * 512
    call = stub.calls[0]
    assert call["model"] == "text-embedding-3-small"
    assert call["dimensions"] == 512
    assert call["input"] == ["hello"]


def test_openai_quota_exhausted_fails_fast(monkeypatch) -> None:
    """The production outage: 429 insufficient_quota must not be retried."""
    err = _APIError(429, "Error code: 429 - {'error': {'code': 'insufficient_quota', ...}}")
    e, stub = _openai(monkeypatch, [err, [[0.1] * 512]])
    with pytest.raises(EmbeddingError, match="429"):
        e.embed("x")
    assert len(stub.calls) == 1


def test_openai_transient_errors_retry_then_raise(monkeypatch) -> None:
    e, stub = _openai(monkeypatch, [_APIError(500, "boom"), _APIError(503, "down"), [[0.2] * 512]])
    assert e.embed("x") == [0.2] * 512
    assert len(stub.calls) == 3
    e2, stub2 = _openai(monkeypatch, [_APIError(502, "bad")] * 4)
    with pytest.raises(EmbeddingError, match="after retries"):
        e2.embed("x")
    assert len(stub2.calls) == 4


def test_openai_bad_request_is_not_retried(monkeypatch) -> None:
    e, stub = _openai(monkeypatch, [_APIError(400, "too long")])
    with pytest.raises(EmbeddingError, match="400"):
        e.embed("x")
    assert len(stub.calls) == 1


def test_classify_failure() -> None:
    assert classify_failure(429, "insufficient_quota") == "fatal"
    assert classify_failure(429, "Rate limit reached") == "retry"
    assert classify_failure(401, "bad key") == "fatal"
    assert classify_failure(500, "oops") == "retry"
    assert classify_failure(None, "ConnectError") == "retry"


# ── HTTP providers (httpx.MockTransport) ─────────────────────────────


def _transport(handler):
    seen: list[httpx.Request] = []

    def _h(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request, json.loads(request.content or b"{}"))

    return httpx.MockTransport(_h), seen


def _openai_shape(vectors):
    return {"object": "list", "data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}


def test_openai_compatible_shape_and_bearer() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[1.0, 0.0, 0.0]] * len(b["input"]))))
    e = hp.OpenAICompatibleEmbedder(model="bge-m3", dim=3, base_url="https://api.deepinfra.com/v1/openai",
                                    api_key="k", transport=t)
    assert e.embed_batch(["a", "b"]) == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    req = seen[0]
    assert str(req.url) == "https://api.deepinfra.com/v1/openai/embeddings"
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body == {"model": "bge-m3", "input": ["a", "b"], "encoding_format": "float"}
    assert e.signature == EmbedSignature("openai_compatible", "bge-m3", 3)


def test_openai_compatible_can_request_dimensions() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[0.0, 1.0]])))
    e = hp.OpenAICompatibleEmbedder(model="m", dim=2, base_url="http://h", request_dim=True, transport=t)
    e.embed("q")
    assert json.loads(seen[0].content)["dimensions"] == 2


def test_tei_native_embed_shape() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=[[0.6, 0.8]] * len(b["inputs"])))
    e = hp.TEIEmbedder(model="granite-97m", dim=2, base_url="http://127.0.0.1:8080", transport=t)
    assert e.embed_batch(["x", "y", "z"]) == [[0.6, 0.8]] * 3
    body = json.loads(seen[0].content)
    assert body == {"inputs": ["x", "y", "z"], "normalize": True, "truncate": True}
    assert str(seen[0].url) == "http://127.0.0.1:8080/embed"
    assert "authorization" not in seen[0].headers  # sidecar, no key


def test_voyage_input_type_differs_for_query_and_document() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[1.0, 0.0]] * len(b["input"]))))
    e = hp.VoyageEmbedder(model="voyage-3.5-lite", dim=2, api_key="vk", request_dim=True, transport=t)
    e.embed("q")
    e.embed_batch(["d1", "d2"])
    q, d = (json.loads(r.content) for r in seen)
    assert q["input_type"] == "query" and d["input_type"] == "document"
    assert q["output_dimension"] == 2
    assert str(seen[0].url) == "https://api.voyageai.com/v1/embeddings"


def test_jina_task_selector() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[1.0, 0.0]] * len(b["input"]))))
    e = hp.JinaEmbedder(model="jina-embeddings-v3", dim=2, api_key="jk", transport=t)
    e.embed("q")
    e.embed_batch(["d"])
    q, d = (json.loads(r.content) for r in seen)
    assert q["task"] == "retrieval.query" and d["task"] == "retrieval.passage"
    assert str(seen[0].url) == "https://api.jina.ai/v1/embeddings"


def test_cohere_v2_embed_shape() -> None:
    def handler(r, b):
        return httpx.Response(200, json={"embeddings": {"float": [[0.0, 1.0]] * len(b["texts"])}})

    t, seen = _transport(handler)
    e = hp.CohereEmbedder(model="embed-v4.0", dim=2, api_key="ck", transport=t)
    assert e.embed("q") == [0.0, 1.0]
    body = json.loads(seen[0].content)
    assert body["texts"] == ["q"] and body["input_type"] == "search_query"
    assert body["embedding_types"] == ["float"]
    e.embed_batch(["d"])
    assert json.loads(seen[1].content)["input_type"] == "search_document"
    assert str(seen[0].url) == "https://api.cohere.com/v2/embed"


def test_http_quota_error_is_fatal_and_not_retried(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    t, seen = _transport(lambda r, b: httpx.Response(429, json={"error": {"code": "insufficient_quota"}}))
    e = hp.OpenAICompatibleEmbedder(model="m", dim=2, base_url="http://h", transport=t)
    with pytest.raises(EmbeddingError, match="429"):
        e.embed("x")
    assert len(seen) == 1


def test_http_5xx_retries_then_raises(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    t, seen = _transport(lambda r, b: httpx.Response(503, text="overloaded"))
    e = hp.TEIEmbedder(model="m", dim=2, base_url="http://h", transport=t)
    with pytest.raises(EmbeddingError, match="4 attempts"):
        e.embed("x")
    assert len(seen) == 4


def test_http_connection_error_surfaces_as_embedding_error(monkeypatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)

    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    e = hp.TEIEmbedder(model="m", dim=2, base_url="http://127.0.0.1:1", transport=httpx.MockTransport(boom))
    with pytest.raises(EmbeddingError, match="ConnectError"):
        e.embed("x")


def test_dim_guard_rejects_wrong_size_unless_matryoshka() -> None:
    t, _ = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[3.0, 4.0, 0.0, 0.0]])))
    strict = hp.OpenAICompatibleEmbedder(model="m", dim=2, base_url="http://h", transport=t)
    with pytest.raises(EmbeddingError, match="4-dim"):
        strict.embed("x")
    sliced = hp.OpenAICompatibleEmbedder(model="m", dim=2, base_url="http://h", transport=t, matryoshka=True)
    v = sliced.embed("x")
    assert v == pytest.approx([0.6, 0.8])  # re-normalised slice
    assert math.isclose(sum(x * x for x in v), 1.0)
    too_short = hp.OpenAICompatibleEmbedder(model="m", dim=8, base_url="http://h", transport=t, matryoshka=True)
    with pytest.raises(EmbeddingError):
        too_short.embed("x")


def test_prefixes_and_batching() -> None:
    t, seen = _transport(lambda r, b: httpx.Response(200, json=_openai_shape([[1.0]] * len(b["input"]))))
    e = hp.OpenAICompatibleEmbedder(model="e5", dim=1, base_url="http://h", transport=t,
                                    query_prefix="query: ", doc_prefix="passage: ", batch_size=2)
    e.embed("q")
    e.embed_batch(["a", "b", "c"])
    bodies = [json.loads(r.content)["input"] for r in seen]
    assert bodies == [["query: q"], ["passage: a", "passage: b"], ["passage: c"]]


def test_factory_builds_each_http_provider(monkeypatch) -> None:
    base = {"AMPERSTAND_EMBED_MODEL": "m", "AMPERSTAND_EMBED_DIM": "4"}
    for provider, extra, cls in [
        ("openai_compatible", {"AMPERSTAND_EMBED_BASE_URL": "http://h"}, hp.OpenAICompatibleEmbedder),
        ("tei", {"AMPERSTAND_EMBED_BASE_URL": "http://h"}, hp.TEIEmbedder),
        ("voyage", {"VOYAGE_API_KEY": "k"}, hp.VoyageEmbedder),
        ("jina", {"JINA_API_KEY": "k"}, hp.JinaEmbedder),
        ("cohere", {"COHERE_API_KEY": "k"}, hp.CohereEmbedder),
    ]:
        e = build_embedder(embed_config_from_env({**base, **extra, "AMPERSTAND_EMBED_PROVIDER": provider}))
        assert isinstance(e, cls), provider
        assert e.signature == EmbedSignature(provider, "m", 4)


# ── optional in-process ONNX (only when a model dir is provided) ─────


def test_onnx_provider_smoke(monkeypatch) -> None:
    import os
    model_dir = os.environ.get("AMPERSTAND_TEST_ONNX_DIR")
    if not model_dir:
        pytest.skip("set AMPERSTAND_TEST_ONNX_DIR=<dir with tokenizer.json + onnx/> to run")
    pytest.importorskip("onnxruntime")
    e = build_embedder_from_env({
        "AMPERSTAND_EMBED_PROVIDER": "onnx", "AMPERSTAND_EMBED_ONNX_DIR": model_dir,
        "AMPERSTAND_EMBED_ONNX_FILE": os.environ.get("AMPERSTAND_TEST_ONNX_FILE", "onnx/model.onnx"),
        "AMPERSTAND_EMBED_DIM": os.environ.get("AMPERSTAND_TEST_ONNX_DIM", "384"),
    })
    v1 = e.embed("the capital of germany")
    v2 = e.embed("berlin is the german capital")
    v3 = e.embed("recipe for sourdough bread")
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert len(v1) == e.dim
    assert math.isclose(cos(v1, v1), 1.0, abs_tol=1e-4)
    assert cos(v1, v2) > cos(v1, v3)
