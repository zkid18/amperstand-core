"""Content extraction from URLs using trafilatura."""

from __future__ import annotations

import html as html_lib
import logging
import re

import httpx
import trafilatura

from amperstand_core import anysite
from amperstand_core.anysite import AnysiteError
from amperstand_core.models import CapturedContent, ContentType

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_YOUTUBE_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"),
    re.compile(r"youtube\.com/embed/([\w-]{11})"),
    re.compile(r"youtube\.com/shorts/([\w-]{11})"),
]


def is_youtube_url(url: str) -> bool:
    return any(p.search(url) for p in _YOUTUBE_PATTERNS)


def extract_youtube_id(url: str) -> str | None:
    for p in _YOUTUBE_PATTERNS:
        m = p.search(url)
        if m:
            return m.group(1)
    return None


# All LinkedIn URLs we route to the dedicated `extract_linkedin` extractor
# instead of the general article extractor. The general one hits the login
# wall and either saves it as garbage (pre-2026-06-14) or now rejects it
# entirely (post the wall-detection commit). The LinkedIn extractor always
# at least gets og:meta back, even for text/image posts where the body is
# JS-rendered and unreachable.
#
# The original pattern set was video-only (it required `ugcPost`/`activity`
# markers, which only appeared on legacy video URL shapes). Modern LinkedIn
# share URLs look like
#   /posts/<author-slug>_<title-slug>-share-<19-digit-activity-id>-<4chars>/
# and don't contain those markers, so we now also match any /posts/ URL.
# `extract_linkedin` already routes correctly internally: video posts go
# through the transcript path, text/image posts fall through to og:meta stub.
_LINKEDIN_PATTERNS = [
    re.compile(r"://(?:www\.)?linkedin\.com/posts/"),
    re.compile(r"://(?:www\.)?linkedin\.com/feed/update/urn:li:(ugcPost|activity):\d+"),
    re.compile(r"://(?:www\.)?linkedin\.com/video/[^/]+"),
    re.compile(r"://(?:www\.)?linkedin\.com/embed/feed/update/urn:li:(ugcPost|activity):\d+"),
]


def is_linkedin_url(url: str) -> bool:
    return any(p.search(url) for p in _LINKEDIN_PATTERNS)


def _fetch_url(url: str) -> str | None:
    """Fetch URL HTML. Free direct tiers first; Anysite only when they fail.

    Order: trafilatura direct → httpx direct → Anysite static parse (1
    credit). The browser-rendered tier (10 credits) is not here — it's the
    retry `extract_article` reaches for once it has seen that the cheaper
    fetch came back empty or walled, so a page that parses fine never pays
    for a render.
    """
    # Tier 1: trafilatura (fast, lightweight, urllib internally)
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return downloaded
    logger.debug("trafilatura failed for %s, trying httpx", url)

    # Tier 2: httpx with browser headers, direct
    try:
        resp = httpx.get(url, headers=_BROWSER_HEADERS, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        if resp.text and len(resp.text.strip()) > 200:
            return resp.text
        logger.debug("httpx (direct) returned thin content for %s", url)
    except httpx.HTTPError as exc:
        logger.debug("httpx (direct) failed for %s: %s", url, exc)

    # Tier 3: Anysite static parse — their egress, not ours
    return _fetch_via_anysite(url)


def _fetch_via_anysite(url: str, *, render: bool = False) -> str | None:
    """Page HTML via Anysite, or None when it's disabled or came back empty.

    `render=True` executes the page in a real browser on their side (10
    credits instead of 1) — for JS-only pages and for walls the static
    fetch can't get past. Anysite's own failures are logged and swallowed
    here: the caller's "failed to fetch/extract" error is the one the user
    should see, and the retry queue's attempt budget bounds how often a
    URL that never works gets tried.
    """
    if not anysite.enabled():
        return None
    try:
        rec = anysite.webparser_render(url) if render else anysite.webparser_parse(url)
    except AnysiteError as exc:
        logger.info("anysite %s failed for %s: %s", "render" if render else "parse", url, exc)
        return None
    html = rec.get("cleaned_html") or ""
    if html and len(html.strip()) > 200:
        logger.info(
            "Fetched %s via anysite %s (%d bytes)",
            url, "render" if render else "parse", len(html),
        )
        return html
    logger.debug("anysite %s returned thin content for %s", "render" if render else "parse", url)
    return None


# HTML markers (widgets/scripts) that indicate the page is an anti-bot
# wall, regardless of the wall's display language. These are concrete
# structural elements, not localized strings.
_CHALLENGE_HTML_MARKERS = (
    "challenges.cloudflare.com",        # Cloudflare Turnstile widget
    "/cdn-cgi/challenge-platform/",     # Cloudflare challenge platform
    "g-recaptcha",                      # Google reCAPTCHA widget
    "h-captcha",                        # hCaptcha widget
    "data-sitekey=",                    # generic captcha sitekey attribute
)

# Title patterns that strongly suggest the page is a wall, not content.
# Substring match is intentional — Cloudflare/Akamai/etc localize the
# trailing text but keep these stems.
_CHALLENGE_TITLE_STEMS = (
    "just a moment",
    "attention required",
    "access denied",
    "ddos protection",
    "verify you are",
    "verifying you are",
    "checking your browser",
    "security check",
    # 404 / not-found pages — many sites serve these with HTTP 200 + a
    # standard error page body, which the extractor used to silently
    # save (friction test: Verge "404 Not Found | The Verge" landed in
    # vault as a real article). Phrases are deliberately specific to
    # avoid false-positives on legit articles whose title contains "404".
    "404 not found",
    "page not found",
    "this page could not be found",
    "page can't be found",
    "page can’t be found",  # smart-quote variant
    "this page isn't available",
    "this page isn’t available",
    # LinkedIn login walls — surfaced 2026-06-14 testing public profile URLs.
    # Some profiles extract fine; others (rate-limited / A/B'd) get the login
    # wall served with title "Sign Up | LinkedIn". Body lands around 450 chars,
    # above the weak-title cutoff, so the title check is what actually catches it.
    "sign up | linkedin",
    "join linkedin",
    # Twitter / X no-JS fallback — title `JavaScript is not available.` is
    # served by twitter.com when the user-agent has JS disabled (which is what
    # trafilatura looks like). Hit during 2026-06-14 testing of bogus tweet URLs.
    "javascript is not available",
)


# Hosts whose HTML is deliberately built for og:meta scraping rather than
# rendered for humans — the anti-bot heuristic must NOT fire on these even
# when their body is short and "challenge-shaped". Order matters less than
# correctness; we match on a `host in url` substring (cheap, no parse).
_OG_META_FALLBACK_HOSTS = (
    "fxtwitter.com",
    "vxtwitter.com",
    "nitter.net",
    "nitter.privacydev.net",
    "twiiit.com",
)

# Markers in title or body that indicate an og:meta-fallback host's OWN 404
# page (i.e. the tweet was deleted / never existed). Only checked under the
# `_OG_META_FALLBACK_HOSTS` bypass — we don't want to false-positive on real
# articles whose body happens to contain "this page doesn't exist."
# Surfaced 2026-06-14 capturing a missing fxtwitter tweet which served
# title "Nothing to see here" + body "Looks like this page doesn't exist.
# Here's a picture of a poodle..."
_OG_META_FALLBACK_NOT_FOUND_MARKERS = (
    "nothing to see here",
    "this page doesn't exist",
    "this page doesn’t exist",  # smart-quote variant
)


# Capture the `<account>` segment of a Twitter-style URL path:
#   twitter.com/karpathy/status/123  →  "karpathy"
#   x.com/karpathy/status/123        →  "karpathy"
#   fxtwitter.com/karpathy/status/123→  "karpathy"
# Used by the silent-corruption guard for fxtwitter URLs whose bogus IDs
# get resolved to a real but unrelated tweet (e.g. /karpathy/status/1234567890
# → a 2009 @pathfinderSport post in Greek).
_TWITTER_URL_ACCOUNT_RE = re.compile(
    r"(?:twitter\.com|x\.com|fxtwitter\.com|vxtwitter\.com|nitter\.[\w.-]+|twiiit\.com)"
    r"/([A-Za-z0-9_]+)/status/",
    re.IGNORECASE,
)

# Capture the actual `@username` from og:title. Twitter and its mirrors all
# format their og:title as something like "Display Name (@username)" or
# "Display Name on X". We pull `@username` when present.
_TWITTER_TITLE_ACCOUNT_RE = re.compile(r"\(@([A-Za-z0-9_]+)\)")


def _twitter_account_from_url(url: str) -> str | None:
    m = _TWITTER_URL_ACCOUNT_RE.search(url)
    return m.group(1).lower() if m else None


def _twitter_account_from_title(title: str | None) -> str | None:
    if not title:
        return None
    m = _TWITTER_TITLE_ACCOUNT_RE.search(title)
    return m.group(1).lower() if m else None


def _looks_like_challenge(
    text: str | None, title: str | None, html: str | None, *, url: str | None = None
) -> bool:
    """Heuristic: does this look like a wall/challenge page rather than an
    article? Three orthogonal signals — any one trips it:

    1. Concrete HTML widget markers (Cloudflare/captcha) anywhere in the
       fetched HTML. Language-independent.
    2. Title is a known wall-page stem ("Just a Moment...", "Access Denied").
    3. Extracted text body is suspiciously short AND the title is missing
       or generic — articles with weak metadata still tend to have body.

    `url` is an optional bypass: if the URL is a known og:meta fallback host
    (fxtwitter, vxtwitter, nitter, ...) we return False unconditionally — those
    hosts deliberately serve short, structured pages that the heuristic would
    otherwise misread as a wall.

    Real articles can be short, so we lean on combinations rather than
    length alone.
    """
    # Trusted og:meta-only hosts: bypass the general heuristic. fxtwitter/
    # nitter return short structured pages by design — the body-length
    # heuristic would otherwise classify their valid responses as walls.
    # We do still check for the host's *own* 404 / deleted-tweet pages,
    # and for "wrong tweet served" silent-corruption cases (fxtwitter
    # resolves bogus IDs to a real-but-unrelated tweet — caught 2026-06-16
    # in queue e2e testing).
    if url:
        url_lower = url.lower()
        for trusted_host in _OG_META_FALLBACK_HOSTS:
            if trusted_host in url_lower:
                norm_title = (title or "").strip().lower()
                body_lower = (text or "").strip().lower()
                for marker in _OG_META_FALLBACK_NOT_FOUND_MARKERS:
                    if marker in norm_title or marker in body_lower:
                        return True
                # Cross-check the URL's intended account against the og:title's
                # actual @username. If they disagree, the host returned a
                # DIFFERENT tweet than the one requested — saving it would
                # attribute someone else's words to the URL's account. Only
                # fires when BOTH are extractable; if the title has no
                # @username we can't compare, so we fall through.
                url_account = _twitter_account_from_url(url)
                title_account = _twitter_account_from_title(title)
                if (
                    url_account is not None
                    and title_account is not None
                    and url_account != title_account
                ):
                    return True
                return False

    if html:
        h = html[:50_000]  # cap scan for huge SPA payloads
        for marker in _CHALLENGE_HTML_MARKERS:
            if marker in h:
                return True

    norm_title = (title or "").strip().lower()
    if norm_title:
        for stem in _CHALLENGE_TITLE_STEMS:
            if stem in norm_title:
                return True

    body = (text or "").strip()
    weak_title = norm_title in {"", "untitled"}
    if weak_title and len(body) < 400:
        return True
    return False


_OG_TITLE_RE = re.compile(
    r'<meta[^>]*\bproperty=["\']og:title["\'][^>]*\bcontent=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_TITLE_REVERSE_RE = re.compile(
    r'<meta[^>]*\bcontent=["\']([^"\']+)["\'][^>]*\bproperty=["\']og:title["\']',
    re.IGNORECASE,
)
_HTML_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _extract_og_title(html: str) -> str | None:
    """Pull <meta property="og:title" content="..."> with attribute order
    tolerance (some sites put content before property). Returns the
    HTML-entity-decoded title or None."""
    if not html:
        return None
    m = _OG_TITLE_RE.search(html) or _OG_TITLE_REVERSE_RE.search(html)
    return html_lib.unescape(m.group(1).strip()) if m else None


def _extract_html_title(html: str) -> str | None:
    if not html:
        return None
    m = _HTML_TITLE_RE.search(html)
    return html_lib.unescape(m.group(1).strip()) if m else None


def _title_appears_in_body(title: str | None, body: str | None) -> bool:
    """Does a meaningful chunk of `title`'s words appear in the body's first
    1500 chars? Used to detect when trafilatura picked a heading from an
    adjacent article on the same page — the wrong-title bug PKM/Researcher
    surfaced on Stratechery, and today's Dan Luu /keyboard-latency/ capture
    landed as 'Appendix: techniques that only work at small scale.'"""
    if not title or not body:
        return False
    title_words = {w.lower() for w in _WORD_RE.findall(title) if len(w) > 2}
    if not title_words:
        return False
    body_words = {w.lower() for w in _WORD_RE.findall(body[:1500])}
    overlap = len(title_words & body_words) / len(title_words)
    return overlap >= 0.5


def _pick_best_title(
    traf_title: str | None,
    og_title: str | None,
    html_title: str | None,
    body: str | None,
) -> str | None:
    """Decide which extracted title to trust. Trafilatura sometimes scrapes a
    heading from an adjacent article on the same page (sidebar listings,
    'related posts' blocks, appendix linkbacks). Cross-check against the
    body — if trafilatura's pick doesn't appear in the body's opening
    1500 chars, prefer og:title (which is page-level metadata and reliably
    matches the URL)."""
    if traf_title and _title_appears_in_body(traf_title, body):
        return traf_title
    if og_title:
        return og_title
    if html_title:
        return html_title
    return traf_title


def _extract_pieces(html: str) -> tuple[str | None, str | None, str | None]:
    """Run trafilatura.extract + extract_metadata once on a downloaded HTML
    blob. Returns (markdown_text, title, author) — any may be None.

    Title pick goes through `_pick_best_title` which cross-checks
    trafilatura's choice against the body and falls back to og:title /
    <title> when trafilatura grabbed a heading from a different article
    on the same page.
    """
    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_images=True,
        include_tables=True,
    )
    metadata = trafilatura.extract_metadata(html)
    traf_title = metadata.title if metadata and metadata.title else None
    author = metadata.author if metadata and metadata.author else None
    title = _pick_best_title(
        traf_title,
        _extract_og_title(html),
        _extract_html_title(html),
        text,
    )
    return text, title, author


def extract_article_from_html(
    url: str, html: str, *, fallback_title: str | None = None
) -> CapturedContent:
    """Run the article extraction pipeline on caller-supplied HTML.

    Used by /capture/html (browser-extension clipper) where the user's
    authenticated browser session has already fetched the page; we don't
    want to refetch from the server. Skips fetch tiers and the proxy
    retry — the caller already got the real page. Challenge detection
    still applies (the HTML may include a captcha widget if the page is
    a paywall stub).
    """
    if not html or not html.strip():
        raise ValueError(f"Empty HTML for {url}")

    text, title, author = _extract_pieces(html)
    if not text:
        raise ValueError(f"Failed to extract content from supplied HTML for: {url}")

    if not title:
        title = fallback_title

    if _looks_like_challenge(text, title, html, url=url):
        raise ValueError(
            f"Supplied HTML for {url} looks like a wall/challenge page; "
            f"capture aborted."
        )

    text = re.sub(r"!\[\]\(\)\s*\|?\s*", "", text)
    text = re.sub(r"^\|?\s*\n", "", text, flags=re.MULTILINE)
    text = text.lstrip("\n")

    return CapturedContent(
        url=url,
        title=title or "Untitled",
        content_markdown=text,
        content_type=ContentType.ARTICLE,
        author=author,
    )


def extract_article(url: str) -> CapturedContent:
    """Fetch a URL and extract its article content as markdown."""
    downloaded = _fetch_url(url)
    text = title = author = None
    if downloaded:
        text, title, author = _extract_pieces(downloaded)

    # Retry with a browser render if the cheap fetch came back empty,
    # trivially short, OR looking like a wall/challenge page. A rendered
    # page executes the site's JS and arrives from Anysite's IP space, which
    # gets the real content where a bare server-side GET is fronted by
    # Cloudflare/captcha/etc.
    needs_retry = (
        not text
        or len(text.strip()) < 200
        or _looks_like_challenge(text, title, downloaded, url=url)
    )
    if needs_retry:
        retry_html = _fetch_via_anysite(url, render=True)
        if retry_html:
            r_text, r_title, r_author = _extract_pieces(retry_html)
            retry_is_better = (
                r_text
                and not _looks_like_challenge(r_text, r_title, retry_html, url=url)
                and len(r_text.strip()) > len((text or "").strip())
            )
            if retry_is_better:
                logger.info("extract via anysite render improved content for %s", url)
                downloaded, text, title, author = retry_html, r_text, r_title, r_author

    if not downloaded:
        raise ValueError(f"Failed to fetch URL: {url}")
    if not text:
        raise ValueError(f"Failed to extract content from: {url}")

    if _looks_like_challenge(text, title, downloaded, url=url):
        raise ValueError(
            f"Anti-bot wall served for {url} — short body / generic title / "
            f"captcha-widget HTML. Capture aborted."
        )

    # Clean up empty image tags and table artifacts
    text = re.sub(r"!\[\]\(\)\s*\|?\s*", "", text)
    text = re.sub(r"^\|?\s*\n", "", text, flags=re.MULTILINE)
    text = text.lstrip("\n")

    return CapturedContent(
        url=url,
        title=title or "Untitled",
        content_markdown=text,
        content_type=ContentType.ARTICLE,
        author=author,
    )
