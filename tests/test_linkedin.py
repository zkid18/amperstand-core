"""LinkedIn capture: URL routing and the Anysite-backed extractor."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from amperstand_core.anysite import AnysiteError
from amperstand_core.audio import AudioError
from amperstand_core.extractor import is_linkedin_url
from amperstand_core.linkedin import extract_linkedin, resolve_post_urn
from amperstand_core.models import ContentType


@pytest.fixture(autouse=True)
def _no_public_page(monkeypatch: pytest.MonkeyPatch):
    """Never touch linkedin.com from the suite; tests that exercise the
    page-based resolution patch this themselves."""
    monkeypatch.setattr("amperstand_core.linkedin._fetch_public_page", lambda url: None)


# ── URL → activity id ────────────────────────────────────────────────

_SHARE_URL = "https://www.linkedin.com/posts/jakesaper_i-have-spent-more-time-writing-about-ai-native-share-7469623060841742336-Ejx0"
_ACTIVITY_ID = "7469780337284304897"
_PAGE_WITH_ATTR = (
    '<meta property="og:url" content="https://www.linkedin.com/posts/jakesaper_slug-activity-'
    + _ACTIVITY_ID + '-wHyq">'
    '<article data-attributed-urn="urn:li:share:7469623060841742336" '
    'data-activity-urn="urn:li:activity:' + _ACTIVITY_ID + '">'
    '<a href="/feed/update/urn:li:activity:111">related</a>'
)


class TestResolvePostUrn:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.linkedin.com/posts/janedoe_topic-activity-9876543210-wHyq", "activity:9876543210"),
            ("https://www.linkedin.com/posts/janedoe_topic-activity-9876543210/", "activity:9876543210"),
            ("https://www.linkedin.com/feed/update/urn:li:activity:9876543210", "activity:9876543210"),
            ("https://www.linkedin.com/feed/update/urn:li:activity:9876543210/?utm=x", "activity:9876543210"),
            ("https://www.linkedin.com/embed/feed/update/urn:li:activity:9876543210", "activity:9876543210"),
        ],
    )
    def test_activity_id_in_url_needs_no_fetch(self, url, expected):
        with patch("amperstand_core.linkedin._fetch_public_page") as fetch:
            assert resolve_post_urn(url) == expected
            fetch.assert_not_called()

    def test_share_url_resolves_via_post_element(self):
        """A share id is not an activity id: Anysite 422s on the share URL
        and 412s on the share number. The public page carries the real one."""
        with patch("amperstand_core.linkedin._fetch_public_page", return_value=_PAGE_WITH_ATTR):
            assert resolve_post_urn(_SHARE_URL) == f"activity:{_ACTIVITY_ID}"

    def test_share_url_falls_back_to_og_url(self):
        page = _PAGE_WITH_ATTR.replace('data-activity-urn="urn:li:activity:' + _ACTIVITY_ID + '"', "")
        with patch("amperstand_core.linkedin._fetch_public_page", return_value=page):
            assert resolve_post_urn(_SHARE_URL) == f"activity:{_ACTIVITY_ID}"

    def test_share_url_falls_back_to_most_frequent_urn(self):
        page = "urn:li:activity:222 urn:li:activity:333 urn:li:activity:333 urn:li:activity:222 urn:li:activity:333"
        with patch("amperstand_core.linkedin._fetch_public_page", return_value=page):
            assert resolve_post_urn(_SHARE_URL) == "activity:333"

    def test_unresolvable_url_passes_through(self):
        """Login wall, 999, network error or a page without URNs: hand the
        URL to Anysite unchanged rather than inventing an id."""
        with patch("amperstand_core.linkedin._fetch_public_page", return_value=None):
            assert resolve_post_urn(_SHARE_URL) == _SHARE_URL
        with patch("amperstand_core.linkedin._fetch_public_page", return_value="<html>authwall</html>"):
            assert resolve_post_urn(_SHARE_URL) == _SHARE_URL

    @patch("amperstand_core.linkedin.anysite.linkedin_post")
    def test_extractor_spends_the_credit_on_the_resolved_id(self, mock_post):
        mock_post.return_value = {"author": {"name": "Jake Saper"}, "text": "hi", "video_url": None}
        with patch("amperstand_core.linkedin._fetch_public_page", return_value=_PAGE_WITH_ATTR):
            doc = extract_linkedin(_SHARE_URL)
        mock_post.assert_called_once_with(f"activity:{_ACTIVITY_ID}")
        assert doc.url == _SHARE_URL


# ── URL detection ────────────────────────────────────────────────────


class TestIsLinkedInUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/posts/johndoe_some-slug-ugcPost-1234567890",
            "https://www.linkedin.com/posts/janedoe_topic-activity-9876543210",
            "https://www.linkedin.com/feed/update/urn:li:ugcPost:1234567890",
            "https://www.linkedin.com/feed/update/urn:li:activity:9876543210",
            "https://www.linkedin.com/video/some-video-id",
            "https://www.linkedin.com/embed/feed/update/urn:li:ugcPost:1234567890",
            "https://www.linkedin.com/embed/feed/update/urn:li:activity:9876543210",
            "https://linkedin.com/posts/user_topic-activity-123456",
            # Modern share URL pattern — no ugcPost/activity marker,
            # uses share-<digits>-<short_code>.
            "https://www.linkedin.com/posts/jakesaper_i-have-spent-more-time-writing-about-ai-native-share-7469623060841742336-Ejx0/",
            "https://www.linkedin.com/posts/someone_some-slug-share-1234567890123456789-aBcD/",
        ],
    )
    def test_valid_linkedin_urls(self, url: str):
        assert is_linkedin_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/in/johndoe",
            "https://www.linkedin.com/company/acme",
            "https://www.linkedin.com/jobs/view/12345",
            "https://www.youtube.com/watch?v=abc123def45",
            "https://www.google.com",
            "https://www.linkedin.com/messaging/thread/123",
            "https://example.com/linkedin.com/posts/fake-ugcPost",
        ],
    )
    def test_non_linkedin_urls(self, url: str):
        assert is_linkedin_url(url) is False


# ── extractor ────────────────────────────────────────────────────────

URL = "https://www.linkedin.com/feed/update/urn:li:activity:7419745318503735296/"
ACTIVITY_ID = "7419745318503735296"

_TEXT_POST = {
    "urn": {"type": "activity", "value": "7469122525881532416"},
    "author": {
        "name": "Olga Maslikhova", "alias": "olga",
        "headline": "Founder at TJC", "url": "https://www.linkedin.com/in/olga",
    },
    "text": "If I were to start a business in Brazil today, here's what I'd do:\n\n1. Talk to customers.\n2. Ship.",
    "video_url": None,
    "images": [],
}

_VIDEO_POST = {
    **_TEXT_POST,
    "author": {"name": "Joe Rhew", "headline": "Applied AI in GTM"},
    "text": "Every time you use ChatGPT or Claude, you're starting from scratch.\n\nCopy. Paste. Repeat.",
    "video_url": "https://dms.licdn.com/playlist/vid/v2/D5605AQHGRLQ62vtZnA/mp4-720p-30fp-crf28/0/1769005122566",
}


class TestTextPost:
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_TEXT_POST)
    def test_full_text_not_a_teaser(self, mock_post):
        doc = extract_linkedin(URL)

        assert doc.url == URL
        assert doc.content_type == ContentType.ARTICLE
        assert doc.author == "Olga Maslikhova"
        assert doc.title == "Olga Maslikhova on LinkedIn: If I were to start a business in Brazil today, here's what I'd do:"
        assert "**Author**: Olga Maslikhova — Founder at TJC" in doc.content_markdown
        # The whole post, not the 150-char og:description.
        assert "2. Ship." in doc.content_markdown
        assert "## Transcript" not in doc.content_markdown
        # The credit is spent on the activity id, never the raw URL.
        mock_post.assert_called_once_with(f"activity:{ACTIVITY_ID}")

    @patch("amperstand_core.linkedin.anysite.linkedin_post")
    def test_long_first_line_is_trimmed_for_title(self, mock_post):
        mock_post.return_value = {**_TEXT_POST, "text": "word " * 60}
        doc = extract_linkedin(URL)
        assert doc.title.endswith("…")
        assert len(doc.title) < 120

    @patch("amperstand_core.linkedin.anysite.linkedin_post")
    def test_empty_post_still_saves(self, mock_post):
        mock_post.return_value = {"author": None, "text": None, "video_url": None}
        doc = extract_linkedin(URL)
        assert doc.title == "LinkedIn post"
        assert "*Post has no text.*" in doc.content_markdown

    @patch("amperstand_core.linkedin.anysite.linkedin_post")
    def test_anysite_failure_propagates(self, mock_post):
        """A 412 means the post is gone. The server turns this into a 422 and
        the retry queue must NOT spend attempts on it — so it has to surface
        as the AnysiteError itself, retryable flag intact."""
        mock_post.side_effect = AnysiteError("Anysite /api/linkedin/post → 412: Post not found", status=412, retryable=False)
        with pytest.raises(AnysiteError) as exc:
            extract_linkedin(URL)
        assert exc.value.retryable is False


class TestVideoPost:
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_VIDEO_POST)
    def test_transcription_off_keeps_text_and_flags_video(self, mock_post, monkeypatch):
        monkeypatch.delenv("AMPERSTAND_AUDIO_TRANSCRIPTION", raising=False)
        monkeypatch.delenv("AMPERSTAND_YOUTUBE_AUDIO_FALLBACK", raising=False)

        doc = extract_linkedin(URL)

        assert doc.content_type == ContentType.VIDEO
        assert "Copy. Paste. Repeat." in doc.content_markdown
        assert "*Includes a video (not transcribed).*" in doc.content_markdown
        assert "## Transcript" not in doc.content_markdown

    @patch("amperstand_core.linkedin.transcribe_audio_file", return_value="Hello from the video.")
    @patch("amperstand_core.linkedin.download_media", return_value=1234)
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_VIDEO_POST)
    def test_transcribed_when_enabled(self, mock_post, mock_dl, mock_whisper, monkeypatch):
        monkeypatch.setenv("AMPERSTAND_AUDIO_TRANSCRIPTION", "1")

        doc = extract_linkedin(URL)

        assert doc.content_type == ContentType.VIDEO
        assert "## Transcript" in doc.content_markdown
        assert "Hello from the video." in doc.content_markdown
        assert mock_dl.call_args.args[0] == _VIDEO_POST["video_url"]

    @patch("amperstand_core.linkedin.transcribe_audio_file")
    @patch("amperstand_core.linkedin.download_media", return_value=1234)
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_VIDEO_POST)
    def test_legacy_youtube_env_name_still_enables(self, mock_post, mock_dl, mock_whisper, monkeypatch):
        monkeypatch.delenv("AMPERSTAND_AUDIO_TRANSCRIPTION", raising=False)
        monkeypatch.setenv("AMPERSTAND_YOUTUBE_AUDIO_FALLBACK", "1")
        mock_whisper.return_value = "legacy ok"
        assert "legacy ok" in extract_linkedin(URL).content_markdown

    @patch("amperstand_core.linkedin.download_media", side_effect=AudioError("media exceeds 25MB Whisper limit"))
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_VIDEO_POST)
    def test_oversized_video_degrades_to_text(self, mock_post, mock_dl, monkeypatch):
        monkeypatch.setenv("AMPERSTAND_AUDIO_TRANSCRIPTION", "1")

        doc = extract_linkedin(URL)

        assert doc.content_type == ContentType.VIDEO
        assert "Copy. Paste. Repeat." in doc.content_markdown
        assert "not transcribed" in doc.content_markdown

    @patch("amperstand_core.linkedin.transcribe_audio_file", side_effect=RuntimeError("openai down"))
    @patch("amperstand_core.linkedin.download_media", return_value=1234)
    @patch("amperstand_core.linkedin.anysite.linkedin_post", return_value=_VIDEO_POST)
    def test_unexpected_whisper_crash_does_not_lose_the_post(self, mock_post, mock_dl, mock_whisper, monkeypatch):
        monkeypatch.setenv("AMPERSTAND_AUDIO_TRANSCRIPTION", "1")
        doc = extract_linkedin(URL)
        assert "Copy. Paste. Repeat." in doc.content_markdown
