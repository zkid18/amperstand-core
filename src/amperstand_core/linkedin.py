"""LinkedIn post capture via Anysite, with optional video transcription.

Anysite returns the full post text, which is what a logged-out fetch never
gets — LinkedIn's og:description is a ~150-character teaser and the body is
JS-rendered behind a login wall. Video posts additionally carry a signed
CDN URL for the mp4; when transcription is enabled that file goes through
Whisper and the transcript is appended to the doc.

Anysite identifies a post by its *activity* id. Only some LinkedIn URLs
carry that id; the modern `/posts/…-share-<id>-<code>` form carries a share
id instead, and Anysite answers 422 for it. `resolve_post_urn` turns every
supported URL shape into `activity:<id>` before the credit is spent.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections import Counter
from pathlib import Path

import httpx

from amperstand_core import anysite
from amperstand_core.audio import (
    AudioError,
    audio_transcription_enabled,
    download_media,
    transcribe_audio_file,
)
from amperstand_core.extractor import _BROWSER_HEADERS
from amperstand_core.models import CapturedContent, ContentType

logger = logging.getLogger(__name__)

_TITLE_CHARS = 80

# `…-activity-<id>` in a /posts/ slug, or `urn:li:activity:<id>` in a
# /feed/update/ URL. The id is at least 10 digits and ends the segment.
_ACTIVITY_IN_URL = re.compile(r"activity[-:](\d{10,})(?=$|[-/?#])")
# On the public post page the post element carries both URNs; the
# activity one is the handle Anysite resolves.
_ACTIVITY_ATTR = re.compile(r'data-activity-urn="urn:li:activity:(\d+)"')
_ACTIVITY_OG_URL = re.compile(r'property="og:url"\s+content="[^"]*-activity-(\d+)')
_ACTIVITY_ANY = re.compile(r"urn:li:activity:(\d+)")
_PAGE_TIMEOUT_S = 20.0


def resolve_post_urn(url: str) -> str:
    """Map a LinkedIn post URL to the `activity:<id>` handle Anysite resolves.

    URLs that already carry the activity id are rewritten directly. Share,
    ugcPost and /video/ URLs carry a different id, so the public post page
    (served without login) is fetched once and the activity URN is read from
    the post element's `data-activity-urn`, then from `og:url`, then from the
    most frequent activity URN on the page. When nothing resolves, the URL
    is passed through unchanged so Anysite gets its own chance at it.
    """
    m = _ACTIVITY_IN_URL.search(url)
    if m:
        return f"activity:{m.group(1)}"
    html = _fetch_public_page(url)
    if html:
        m = _ACTIVITY_ATTR.search(html) or _ACTIVITY_OG_URL.search(html)
        if m:
            return f"activity:{m.group(1)}"
        ids = _ACTIVITY_ANY.findall(html)
        if ids:
            return f"activity:{Counter(ids).most_common(1)[0][0]}"
    logger.info("no activity id resolved for %s; passing the URL to Anysite", url)
    return url


def _fetch_public_page(url: str) -> str | None:
    """One anonymous GET of the post page; None on any failure or non-200."""
    try:
        resp = httpx.get(
            url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=_PAGE_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        logger.debug("public page fetch failed for %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        logger.debug("public page fetch for %s returned %d", url, resp.status_code)
        return None
    return resp.text


def extract_linkedin(url: str) -> CapturedContent:
    post = anysite.linkedin_post(resolve_post_urn(url))

    author_rec = post.get("author") or {}
    author = (author_rec.get("name") or "").strip() or None
    headline = (author_rec.get("headline") or "").strip()
    text = (post.get("text") or "").strip()
    video_url = post.get("video_url")

    lines: list[str] = []
    if author:
        lines.append(f"**Author**: {author}" + (f" — {headline}" if headline else ""))
        lines.append("")
    lines.append(text if text else "*Post has no text.*")

    content_type = ContentType.ARTICLE
    if video_url:
        content_type = ContentType.VIDEO
        transcript = _transcribe(video_url, url)
        if transcript:
            lines += ["", "## Transcript", "", transcript]
        else:
            lines += ["", "*Includes a video (not transcribed).*"]

    return CapturedContent(
        url=url,
        title=_title(text, author),
        content_markdown="\n".join(lines),
        content_type=content_type,
        author=author,
    )


def _title(text: str, author: str | None) -> str:
    """LinkedIn posts have no title. Mirror the shape LinkedIn itself uses
    for og:title — author, then the opening of the post — so docs sort and
    read the way the ones captured before this backend did."""
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if len(first) > _TITLE_CHARS:
        first = first[:_TITLE_CHARS].rsplit(" ", 1)[0] + "…"
    if author and first:
        return f"{author} on LinkedIn: {first}"
    if first:
        return first
    return f"{author} on LinkedIn" if author else "LinkedIn post"


def _transcribe(video_url: str, post_url: str) -> str | None:
    """Best-effort Whisper transcript of the post's video. Never raises —
    the post text is the capture; the transcript is a bonus."""
    if not audio_transcription_enabled():
        logger.info("LinkedIn video on %s left untranscribed (transcription disabled)", post_url)
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="amp-li-video-") as tmp:
            path = Path(tmp) / "video.mp4"
            size = download_media(video_url, path)
            text = transcribe_audio_file(path).strip()
            logger.info(
                "LinkedIn video transcribed for %s (%d chars from %d bytes)",
                post_url, len(text), size,
            )
            return text or None
    except AudioError as exc:
        logger.warning("LinkedIn video transcription skipped for %s: %s", post_url, exc)
    except Exception:  # noqa: BLE001
        logger.exception("LinkedIn video transcription crashed for %s", post_url)
    return None
