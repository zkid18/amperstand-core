"""YouTube transcript and metadata extraction.

Metadata (title, channel) comes from YouTube's public oembed endpoint, which
answers datacenter IPs without complaint. Captions come from Anysite, which
owns the problem of YouTube blocking server IPs — a caption library or media
downloader running on the box gets blocked, and nothing installed locally
can change that.

When a video has no captions the function raises YouTubeTranscriptUnavailable
carrying whatever metadata it did get, so the clipper path can still build a
stub from the watch page's own og:meta.
"""

from __future__ import annotations

import html as _htmllib
import logging
import re
import urllib.parse

import httpx

from amperstand_core import anysite
from amperstand_core.anysite import AnysiteError
from amperstand_core.models import CapturedContent, ContentType

logger = logging.getLogger(__name__)

OEMBED_TIMEOUT = 10
OEMBED_URL = "https://www.youtube.com/oembed"

# Language asked for explicitly when the video record's own caption track is
# empty. Auto-captions exist for nearly every English video.
FALLBACK_CAPTION_LANG = "en"


class YouTubeTranscriptUnavailable(ValueError):
    """Raised when no usable transcript could be fetched for a YouTube URL.

    Carries the partial metadata we did fetch (title, channel) so callers
    can still construct a stub if they choose. Used by the server's
    /capture/html dispatch to fall back to caller-supplied HTML — most
    YouTube watch pages render the description into the DOM, which is
    still useful even without a transcript.
    """

    def __init__(self, message: str, *, title: str, channel: str, channel_url: str | None) -> None:
        super().__init__(message)
        self.title = title
        self.channel = channel
        self.channel_url = channel_url


def extract_youtube(url: str) -> CapturedContent:
    """Extract transcript + metadata from a YouTube URL.

    Raises YouTubeTranscriptUnavailable when no transcript could be fetched
    — saving a stub doc that just says "no transcript" is worse than
    failing, since callers can choose a richer fallback (page HTML).
    """
    video_id = _video_id(url) or ""
    meta = _oembed(url)

    title = meta.get("title") or "Untitled Video"
    channel = meta.get("author_name") or "Unknown"
    channel_url = meta.get("author_url")

    record, transcript, lang, reason = _captions(video_id)
    if record:
        channel = channel if channel != "Unknown" else (record.get("author") or "Unknown")

    if not transcript:
        raise YouTubeTranscriptUnavailable(
            f"No transcript for {url} — {reason}",
            title=title,
            channel=channel,
            channel_url=channel_url,
        )

    lines: list[str] = [f"**Channel**: {channel}"]
    if channel_url:
        lines.append(f"**Channel URL**: {channel_url}")
    duration = (record or {}).get("duration_seconds")
    if isinstance(duration, (int, float)) and duration > 0:
        lines.append(f"**Duration**: {int(duration) // 60} min")
    lines.append("")
    description = ((record or {}).get("description") or "").strip()
    if description:
        lines += ["## Description", "", description, ""]
    lines.append(f"## Transcript ({lang})" if lang else "## Transcript")
    lines.append("")
    lines.append(transcript)

    return CapturedContent(
        url=url,
        title=title,
        content_markdown="\n".join(lines),
        content_type=ContentType.VIDEO,
        author=channel,
    )


def _captions(video_id: str) -> tuple[dict | None, str | None, str | None, str]:
    """Fetch the video record and a caption track via Anysite.

    Returns (record, transcript, language, reason). The record's own
    `subtitles` field covers most videos in one call; an explicit request
    for the fallback language is the second and last credit spent. A 412
    means "no track", which is a fact about the video and not worth
    retrying; any other Anysite failure propagates so the retry queue can
    try again later.
    """
    if not video_id:
        return None, None, None, "could not parse a video id from the URL"
    if not anysite.enabled():
        return None, None, None, f"{anysite.TOKEN_ENV} is not set on this server"

    record: dict | None = None
    try:
        record = anysite.youtube_video(video_id)
    except AnysiteError as exc:
        if exc.status != 412:
            raise
        return None, None, None, "video not found or unavailable"

    text = _subtitle_text(record)
    if text:
        return record, text, _language_of(record), None

    try:
        track = anysite.youtube_subtitles(video_id, lang=FALLBACK_CAPTION_LANG)
    except AnysiteError as exc:
        if exc.status != 412:
            raise
        return record, None, None, "captions disabled or no track in our languages"

    text = _subtitle_text(track)
    if text:
        return record, text, _language_of(track) or FALLBACK_CAPTION_LANG, None
    return record, None, None, "captions disabled or no track in our languages"


def _subtitle_text(record: dict) -> str:
    """Flatten whichever caption shape a record carries into one line of text.

    A track is `{text, language, subtitle_count, subtitles[]}`: the
    subtitles endpoint returns one at top level, and the video endpoint
    nests one under its own `subtitles` key. A string, a list of strings or
    a list of timed lines is also accepted, so a change on their side
    degrades to "no captions" rather than a crash.
    """
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return " ".join(text.split())
    subs = record.get("subtitles")
    if isinstance(subs, dict):
        return _subtitle_text(subs)
    if isinstance(subs, str):
        return " ".join(subs.split())
    if isinstance(subs, list):
        parts = []
        for s in subs:
            piece = s.get("text") if isinstance(s, dict) else s
            if isinstance(piece, str) and piece.strip():
                parts.append(" ".join(piece.split()))
        return " ".join(parts)
    return ""


def _language_of(record: dict) -> str | None:
    """Language of the track, whether the record is a track or nests one."""
    lang = record.get("language")
    if not (isinstance(lang, str) and lang.strip()):
        nested = record.get("subtitles")
        lang = nested.get("language") if isinstance(nested, dict) else None
    return lang if isinstance(lang, str) and lang.strip() else None


def youtube_stub_from_html(
    url: str,
    page_html: str,
    fallback: YouTubeTranscriptUnavailable | None = None,
) -> CapturedContent:
    """Build a YouTube CapturedContent from caller-supplied watch-page HTML.

    For the clipper path when no transcript is available. trafilatura
    extracts the page chrome (sidebar, comments) on YouTube — not useful.
    Instead, pull the canonical bits straight from <meta property="og:*">
    which YouTube reliably sets server-side. Falls back to whatever
    metadata oembed already gave us (carried on the exception).
    """
    title = _meta_content(page_html, "og:title") or (fallback.title if fallback else None) or "Untitled Video"
    description = _meta_content(page_html, "og:description") or _meta_content(page_html, "description") or ""
    channel = (fallback.channel if fallback else None) or "Unknown"
    channel_url = fallback.channel_url if fallback else None

    lines: list[str] = [f"**Channel**: {channel}"]
    if channel_url:
        lines.append(f"**Channel URL**: {channel_url}")
    lines.append("")
    lines.append("*Transcript unavailable for this video.*")
    if description.strip():
        lines.append("")
        lines.append("## Description")
        lines.append("")
        lines.append(description.strip())

    return CapturedContent(
        url=url,
        title=title,
        content_markdown="\n".join(lines),
        content_type=ContentType.VIDEO,
        author=channel,
    )


def _meta_content(html: str, name: str) -> str | None:
    """Extract the content of a <meta name|property="X"> tag. None if absent.

    Hand-rolled rather than via BeautifulSoup to avoid pulling in another
    dep just for two attributes. Quote handling: capture group [^Q]+ uses
    backreference \\1 to the opening quote so apostrophes inside a
    double-quoted value (and vice-versa) don't truncate the match.
    """
    if not html:
        return None
    n = re.escape(name)
    patterns = (
        # <meta property="X" content="Y">
        rf'<meta\b[^>]*?\b(?:property|name)\s*=\s*(["\']){n}\1[^>]*?\bcontent\s*=\s*(["\'])(.*?)\2',
        # <meta content="Y" property="X">  (attribute order reversed)
        rf'<meta\b[^>]*?\bcontent\s*=\s*(["\'])(.*?)\1[^>]*?\b(?:property|name)\s*=\s*(["\']){n}\3',
    )
    for pattern in patterns:
        m = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if m:
            # Group index of the captured content varies per pattern.
            # First pattern: groups (open_q, close_q, content) → index 3.
            # Second pattern: groups (open_q, content, close_q) → index 2.
            value = m.group(3) if pattern is patterns[0] else m.group(2)
            return _htmllib.unescape(value)
    return None


def _video_id(url: str) -> str | None:
    """Pull the video id out of a youtube.com or youtu.be URL."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/", 1)[0] or None
    if "youtube.com" in host:
        qs = urllib.parse.parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # /shorts/<id>, /embed/<id>, /live/<id>
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
            return parts[1]
    return None


def _oembed(url: str) -> dict:
    """Fetch oembed metadata. Returns {} on failure (don't raise)."""
    try:
        with httpx.Client(
            timeout=OEMBED_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 amperstand"},
        ) as client:
            r = client.get(OEMBED_URL, params={"url": url, "format": "json"})
        if r.status_code != 200:
            return {}
        return r.json()
    except (httpx.HTTPError, ValueError):
        return {}
