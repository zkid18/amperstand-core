"""extract_article's fetch chain: free direct tiers, then Anysite parse, then render."""

from __future__ import annotations

import httpx
import pytest

from amperstand_core import extractor
from amperstand_core.anysite import AnysiteError

ARTICLE_HTML = """<!DOCTYPE html><html><head><title>An Essay on Phonk</title>
<meta property="og:title" content="An Essay on Phonk" /><meta name="author" content="Some Author" /></head>
<body><article><h1>An Essay on Phonk</h1>
<p>Phonk is a subgenre of hip hop and trap music that emerged in the early 2010s, drawing
inspiration from 1990s Memphis rap. The genre is characterized by its use of cowbell samples,
distorted vocals, and lo-fi aesthetics. It has gained significant popularity through TikTok
in recent years.</p>
<p>Modern phonk has evolved into multiple subgenres including drift phonk, which pairs the
sound with car culture videos, and house phonk, which borrows four-on-the-floor rhythms.
Producers typically work in bedrooms with cheap samplers and free software.</p>
</article></body></html>"""

WALL_HTML = """<html><head><title>Just a moment...</title>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js"></script></head>
<body><p>Checking your browser before accessing the site.</p></body></html>"""


@pytest.fixture
def no_direct(monkeypatch: pytest.MonkeyPatch):
    """Both free tiers fail, as they do for a server that the site blocks."""
    monkeypatch.setattr(extractor.trafilatura, "fetch_url", lambda url: None)

    def blocked(*a, **k):
        raise httpx.ConnectError("blocked")
    monkeypatch.setattr(extractor.httpx, "get", blocked)


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANYSITE_ACCESS_TOKEN", "tok")


def test_static_parse_is_tried_before_render(no_direct, token, monkeypatch):
    calls = []
    monkeypatch.setattr(extractor.anysite, "webparser_parse", lambda url: calls.append("parse") or {"cleaned_html": ARTICLE_HTML})
    monkeypatch.setattr(extractor.anysite, "webparser_render", lambda url: calls.append("render") or {})

    doc = extractor.extract_article("https://example.com/phonk")

    assert doc.title == "An Essay on Phonk"
    assert "Memphis rap" in doc.content_markdown
    assert calls == ["parse"]


def test_walled_direct_fetch_is_retried_with_render(token, monkeypatch):
    monkeypatch.setattr(extractor.trafilatura, "fetch_url", lambda url: WALL_HTML)
    monkeypatch.setattr(extractor.anysite, "webparser_render", lambda url: {"cleaned_html": ARTICLE_HTML})

    doc = extractor.extract_article("https://example.com/phonk")

    assert doc.title == "An Essay on Phonk"


def test_empty_parse_falls_through_to_render(no_direct, token, monkeypatch):
    monkeypatch.setattr(extractor.anysite, "webparser_parse", lambda url: {"cleaned_html": ""})
    monkeypatch.setattr(extractor.anysite, "webparser_render", lambda url: {"cleaned_html": ARTICLE_HTML})

    doc = extractor.extract_article("https://example.com/spa")

    assert "cowbell" in doc.content_markdown


def test_render_that_is_still_a_wall_fails_cleanly(no_direct, token, monkeypatch):
    monkeypatch.setattr(extractor.anysite, "webparser_parse", lambda url: {"cleaned_html": ""})
    monkeypatch.setattr(extractor.anysite, "webparser_render", lambda url: {"cleaned_html": WALL_HTML})

    with pytest.raises(ValueError) as exc:
        extractor.extract_article("https://example.com/walled")
    assert "Failed to fetch URL" in str(exc.value) or "wall" in str(exc.value).lower()


def test_anysite_errors_are_swallowed_into_fetch_failure(no_direct, token, monkeypatch):
    def nope(url):
        raise AnysiteError("412", status=412, retryable=False)
    monkeypatch.setattr(extractor.anysite, "webparser_parse", nope)
    monkeypatch.setattr(extractor.anysite, "webparser_render", nope)

    with pytest.raises(ValueError, match="Failed to fetch URL"):
        extractor.extract_article("https://example.com/gone")


def test_without_token_anysite_tiers_are_skipped(no_direct, monkeypatch):
    monkeypatch.delenv("ANYSITE_ACCESS_TOKEN", raising=False)
    called = []
    monkeypatch.setattr(extractor.anysite, "webparser_parse", lambda url: called.append(1))
    monkeypatch.setattr(extractor.anysite, "webparser_render", lambda url: called.append(1))

    with pytest.raises(ValueError, match="Failed to fetch URL"):
        extractor.extract_article("https://example.com/x")
    assert called == []


def test_direct_success_never_touches_anysite(token, monkeypatch):
    monkeypatch.setattr(extractor.trafilatura, "fetch_url", lambda url: ARTICLE_HTML)
    def boom(url):
        raise AssertionError("anysite must not be called")
    monkeypatch.setattr(extractor.anysite, "webparser_parse", boom)
    monkeypatch.setattr(extractor.anysite, "webparser_render", boom)

    assert extractor.extract_article("https://example.com/phonk").title == "An Essay on Phonk"
