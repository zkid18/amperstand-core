"""Anysite HTTP client — the one fetch path for platforms that block servers.

LinkedIn, YouTube, Reddit and anti-bot article pages used to be handled on
the box by a headless Chromium, yt-dlp and a residential proxy. All three are
fast-moving external components that YouTube and LinkedIn break on a weekly
cadence, and on a small droplet nobody re-installs them — each one rotted
silently for months. Anysite (anysite.io) keeps those scrapers working as its
business, so the capture paths that need them call it over plain HTTPS.

Config: ANYSITE_ACCESS_TOKEN enables it. Every call costs credits — 1 for a
post, a caption track or a static page, 10 for a browser-rendered page — so
the extractors reach for it only where the free direct fetch can't get the
content.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.anysite.io"
TOKEN_ENV = "ANYSITE_ACCESS_TOKEN"
BASE_URL_ENV = "ANYSITE_BASE_URL"

# Structured endpoints and static parses answer in seconds. A browser
# render of a heavy page can take a while.
_TIMEOUT_S = 60.0
_RENDER_TIMEOUT_S = 120.0

# Statuses that describe the request or the URL rather than the moment:
# a post that doesn't exist (412), a rejected token (401), a document
# instead of a page (415), a malformed body (422). Retrying won't change
# any of them, so the retry queue shouldn't spend attempts on them.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 412, 415, 422})


class AnysiteError(RuntimeError):
    """A call to Anysite failed.

    `retryable` tells the retry queue whether another attempt could help:
    an upstream hiccup, a rate limit or a network error may clear on its
    own; a 412 or a bad token won't.
    """

    def __init__(
        self, message: str, *, status: int | None = None, retryable: bool,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def token() -> str | None:
    return os.environ.get(TOKEN_ENV, "").strip() or None


def enabled() -> bool:
    return token() is not None


def _base_url() -> str:
    return os.environ.get(BASE_URL_ENV, "").strip().rstrip("/") or DEFAULT_BASE_URL


def _unwrap(payload: Any) -> dict:
    """Reduce a response to the single record that was asked for.

    Endpoints answer with the record itself, a list of records, or an
    `{items: [...]}` envelope. Every extractor here wants one record, so the
    three shapes are normalised in one place.
    """
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        payload = payload["items"]
    if isinstance(payload, list):
        if not payload:
            raise AnysiteError("Anysite returned no records", retryable=False)
        payload = payload[0]
    if not isinstance(payload, dict):
        raise AnysiteError(
            f"Anysite returned an unexpected payload: {type(payload).__name__}",
            retryable=False,
        )
    return payload


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("detail"):
            return str(body["detail"])[:300]
    except ValueError:
        pass
    return (resp.text or "").strip()[:300]


def _post(path: str, body: dict[str, Any], *, timeout: float = _TIMEOUT_S) -> dict:
    tok = token()
    if not tok:
        raise AnysiteError(
            f"{TOKEN_ENV} is not set; Anysite-backed capture is disabled",
            retryable=False,
        )
    url = f"{_base_url()}{path}"
    try:
        resp = httpx.post(url, json=body, headers={"access-token": tok}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise AnysiteError(
            f"Anysite request failed for {path}: {exc}", retryable=True,
        ) from exc

    if resp.status_code >= 400:
        raise AnysiteError(
            f"Anysite {path} → {resp.status_code}: {_detail(resp)}",
            status=resp.status_code,
            retryable=resp.status_code not in _PERMANENT_STATUSES,
        )
    try:
        return _unwrap(resp.json())
    except ValueError as exc:
        raise AnysiteError(
            f"Anysite {path} returned non-JSON", status=resp.status_code, retryable=True,
        ) from exc


# ── endpoints ───────────────────────────────────────────────────────
# Names mirror Anysite's paths so the credit catalog is easy to cross-check.

def linkedin_post(url: str) -> dict:
    """Full post record: text, author{name,url,headline}, video_url, images…"""
    return _post("/api/linkedin/post", {"urn": url})


def youtube_video(video: str) -> dict:
    """Video record: author, description, duration_seconds, subtitles."""
    return _post("/api/youtube/video", {"video": video})


def youtube_subtitles(video: str, *, lang: str = "en") -> dict:
    """One caption track: {text, language, subtitle_count, subtitles[]}."""
    return _post("/api/youtube/video/subtitles", {"video": video, "lang": lang})


def reddit_post(url: str) -> dict:
    return _post("/api/reddit/posts", {"post_url": url})


def webparser_parse(url: str) -> dict:
    """Static fetch, cleaned. Full document so title/author metadata survive."""
    return _post("/api/webparser/parse", {"url": url, "return_full_html": True})


def webparser_render(url: str) -> dict:
    """Same, executed in a real browser on Anysite's side. 10 credits."""
    return _post(
        "/api/webparser/render",
        {"url": url, "return_full_html": True},
        timeout=_RENDER_TIMEOUT_S,
    )
