"""Anysite client: auth, response unwrapping, and which failures are worth retrying."""

from __future__ import annotations

import json

import httpx
import pytest

from amperstand_core import anysite
from amperstand_core.anysite import AnysiteError


class _FakeResponse:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(anysite.TOKEN_ENV, "tok-123")
    return "tok-123"


def _stub_post(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse, seen: dict):
    def fake_post(url, *, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout)
        return response
    monkeypatch.setattr(anysite.httpx, "post", fake_post)


class TestAuth:
    def test_disabled_without_token(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(anysite.TOKEN_ENV, raising=False)
        assert anysite.enabled() is False
        with pytest.raises(AnysiteError) as exc:
            anysite.linkedin_post("https://www.linkedin.com/posts/x")
        assert exc.value.retryable is False

    def test_sends_access_token_header_not_bearer(
        self, monkeypatch: pytest.MonkeyPatch, token: str,
    ):
        seen: dict = {}
        _stub_post(monkeypatch, _FakeResponse(200, {"text": "hi"}), seen)
        anysite.linkedin_post("https://www.linkedin.com/posts/x")
        assert seen["headers"] == {"access-token": token}
        assert seen["url"] == "https://api.anysite.io/api/linkedin/post"
        assert seen["body"] == {"urn": "https://www.linkedin.com/posts/x"}

    def test_base_url_override(self, monkeypatch: pytest.MonkeyPatch, token: str):
        monkeypatch.setenv(anysite.BASE_URL_ENV, "http://localhost:9999/")
        seen: dict = {}
        _stub_post(monkeypatch, _FakeResponse(200, {}), seen)
        anysite.reddit_post("/r/x/comments/1/")
        assert seen["url"] == "http://localhost:9999/api/reddit/posts"


class TestUnwrap:
    """The API answers with a record, a list, or an {items: [...]} envelope."""

    def test_bare_record(self):
        assert anysite._unwrap({"text": "a"}) == {"text": "a"}

    def test_list_takes_first(self):
        assert anysite._unwrap([{"text": "a"}, {"text": "b"}]) == {"text": "a"}

    def test_items_envelope(self):
        assert anysite._unwrap({"items": [{"text": "a"}], "total": 1}) == {"text": "a"}

    def test_empty_list_is_an_error(self):
        with pytest.raises(AnysiteError) as exc:
            anysite._unwrap([])
        assert exc.value.retryable is False

    def test_scalar_is_an_error(self):
        with pytest.raises(AnysiteError):
            anysite._unwrap("nope")


class TestErrorMapping:
    @pytest.mark.parametrize("status", [401, 412, 415, 422])
    def test_permanent_statuses_are_not_retryable(
        self, monkeypatch: pytest.MonkeyPatch, token: str, status: int,
    ):
        _stub_post(monkeypatch, _FakeResponse(status, {"detail": "nope"}), {})
        with pytest.raises(AnysiteError) as exc:
            anysite.youtube_subtitles("abc")
        assert exc.value.status == status
        assert exc.value.retryable is False
        assert "nope" in str(exc.value)

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_transient_statuses_are_retryable(
        self, monkeypatch: pytest.MonkeyPatch, token: str, status: int,
    ):
        _stub_post(monkeypatch, _FakeResponse(status, "upstream sad"), {})
        with pytest.raises(AnysiteError) as exc:
            anysite.webparser_parse("https://example.com")
        assert exc.value.retryable is True

    def test_network_failure_is_retryable(self, monkeypatch: pytest.MonkeyPatch, token: str):
        def boom(*a, **k):
            raise httpx.ConnectError("no route")
        monkeypatch.setattr(anysite.httpx, "post", boom)
        with pytest.raises(AnysiteError) as exc:
            anysite.webparser_render("https://example.com")
        assert exc.value.retryable is True

    def test_non_json_success_is_retryable(self, monkeypatch: pytest.MonkeyPatch, token: str):
        _stub_post(monkeypatch, _FakeResponse(200, "<html>oops</html>"), {})
        with pytest.raises(AnysiteError) as exc:
            anysite.youtube_video("abc")
        assert exc.value.retryable is True


class TestEndpointShapes:
    def test_render_asks_for_full_html_with_long_timeout(
        self, monkeypatch: pytest.MonkeyPatch, token: str,
    ):
        seen: dict = {}
        _stub_post(monkeypatch, _FakeResponse(200, {"cleaned_html": "<html/>"}), seen)
        anysite.webparser_render("https://example.com/spa")
        assert seen["body"] == {"url": "https://example.com/spa", "return_full_html": True}
        assert seen["timeout"] == anysite._RENDER_TIMEOUT_S

    def test_subtitles_default_language(self, monkeypatch: pytest.MonkeyPatch, token: str):
        seen: dict = {}
        _stub_post(monkeypatch, _FakeResponse(200, {"text": "..."}), seen)
        anysite.youtube_subtitles("dQw4w9WgXcQ")
        assert seen["body"] == {"video": "dQw4w9WgXcQ", "lang": "en"}
