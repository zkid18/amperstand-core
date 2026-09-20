"""YouTube capture: Anysite captions + oembed metadata."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from amperstand_core.anysite import AnysiteError
from amperstand_core.models import ContentType
from amperstand_core.youtube import (
    YouTubeTranscriptUnavailable,
    _language_of,
    _subtitle_text,
    _video_id,
    extract_youtube,
)

URL = "https://youtu.be/brOtbWIViWM"
_OEMBED = {"title": "Brazil's lost computer industry", "author_name": "Asianometry", "author_url": "https://youtube.com/@Asianometry"}
_TIMED_LINES = [
    {"start": 2.6, "duration": 5.1, "text": "in the 1980s Brazil had a large domestic"},
    {"start": 5.7, "duration": 4.5, "text": "computer industry dozens of"},
]
# Live shape of POST /api/youtube/video (checked 2026-09-20): a full track
# nested under `subtitles`.
_VIDEO_WITH_SUBS = {
    "@type": "@youtube_video",
    "id": "brOtbWIViWM", "author": "Asianometry", "duration_seconds": 1404, "view_count": 288196,
    "description": "In the 1980s Brazil had a large domestic computer industry.",
    "subtitles": {
        "@type": "@youtube_subtitle",
        "text": "in the 1980s Brazil had a large domestic computer industry dozens of",
        "subtitle_count": 2, "language": "en", "subtitles": _TIMED_LINES,
    },
}
_TRACK = {"text": "explicit english track text", "language": "en", "subtitle_count": 600, "subtitles": []}


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANYSITE_ACCESS_TOKEN", "tok")


class TestVideoId:
    @pytest.mark.parametrize("url,vid", [
        ("https://youtu.be/brOtbWIViWM?is=ZrEgZBomDmLIlcLn", "brOtbWIViWM"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abc123DEF45", "abc123DEF45"),
        ("https://www.youtube.com/live/w0S7nMD1J-I?is=x", "w0S7nMD1J-I"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/watch?v=nope", None),
    ])
    def test_parse(self, url, vid):
        assert _video_id(url) == vid


class TestSubtitleText:
    def test_prefers_joined_text(self):
        assert _subtitle_text({"text": "a  b\nc", "subtitles": [{"text": "zzz"}]}) == "a b c"

    def test_timed_lines(self):
        assert _subtitle_text({"subtitles": _TIMED_LINES}) == "in the 1980s Brazil had a large domestic computer industry dozens of"

    def test_nested_track_on_video_record(self):
        assert _subtitle_text(_VIDEO_WITH_SUBS) == "in the 1980s Brazil had a large domestic computer industry dozens of"
        assert _language_of(_VIDEO_WITH_SUBS) == "en"
        # Nested track without joined text still flattens its timed lines.
        nested = {"subtitles": {"language": "pt", "subtitles": _TIMED_LINES}}
        assert _subtitle_text(nested).startswith("in the 1980s")
        assert _language_of(nested) == "pt"
        assert _language_of({"subtitles": {"text": "x"}}) is None

    def test_string_and_list_of_strings(self):
        assert _subtitle_text({"subtitles": "one  two"}) == "one two"
        assert _subtitle_text({"subtitles": ["one", " two "]}) == "one two"

    def test_missing(self):
        assert _subtitle_text({}) == ""
        assert _subtitle_text({"subtitles": None}) == ""


class TestExtractYoutube:
    @patch("amperstand_core.youtube.anysite.youtube_subtitles")
    @patch("amperstand_core.youtube.anysite.youtube_video", return_value=_VIDEO_WITH_SUBS)
    @patch("amperstand_core.youtube._oembed", return_value=_OEMBED)
    def test_captions_from_video_record_cost_one_call(self, mock_oembed, mock_video, mock_subs):
        doc = extract_youtube(URL)

        assert doc.content_type == ContentType.VIDEO
        assert doc.title == "Brazil's lost computer industry"
        assert doc.author == "Asianometry"
        md = doc.content_markdown
        assert "**Channel URL**: https://youtube.com/@Asianometry" in md
        assert "**Duration**: 23 min" in md
        assert "## Description" in md and "domestic computer industry." in md
        assert "## Transcript" in md and "dozens of" in md
        mock_video.assert_called_once_with("brOtbWIViWM")
        mock_subs.assert_not_called()

    @patch("amperstand_core.youtube.anysite.youtube_subtitles", return_value=_TRACK)
    @patch("amperstand_core.youtube.anysite.youtube_video", return_value={**_VIDEO_WITH_SUBS, "subtitles": []})
    @patch("amperstand_core.youtube._oembed", return_value=_OEMBED)
    def test_falls_back_to_explicit_english_track(self, mock_oembed, mock_video, mock_subs):
        doc = extract_youtube(URL)
        assert "## Transcript (en)" in doc.content_markdown
        assert "explicit english track text" in doc.content_markdown
        mock_subs.assert_called_once_with("brOtbWIViWM", lang="en")

    @patch("amperstand_core.youtube.anysite.youtube_subtitles", side_effect=AnysiteError("412", status=412, retryable=False))
    @patch("amperstand_core.youtube.anysite.youtube_video", return_value={**_VIDEO_WITH_SUBS, "subtitles": None})
    @patch("amperstand_core.youtube._oembed", return_value=_OEMBED)
    def test_no_captions_raises_with_metadata(self, mock_oembed, mock_video, mock_subs):
        with pytest.raises(YouTubeTranscriptUnavailable) as exc:
            extract_youtube(URL)
        assert exc.value.title == "Brazil's lost computer industry"
        assert exc.value.channel == "Asianometry"
        assert "captions disabled" in str(exc.value)

    @patch("amperstand_core.youtube.anysite.youtube_video", side_effect=AnysiteError("412", status=412, retryable=False))
    @patch("amperstand_core.youtube._oembed", return_value={})
    def test_missing_video_uses_placeholders(self, mock_oembed, mock_video):
        with pytest.raises(YouTubeTranscriptUnavailable) as exc:
            extract_youtube(URL)
        assert exc.value.title == "Untitled Video"
        assert "not found" in str(exc.value)

    @patch("amperstand_core.youtube.anysite.youtube_video", side_effect=AnysiteError("503", status=503, retryable=True))
    @patch("amperstand_core.youtube._oembed", return_value=_OEMBED)
    def test_transient_anysite_failure_propagates_for_retry(self, mock_oembed, mock_video):
        with pytest.raises(AnysiteError) as exc:
            extract_youtube(URL)
        assert exc.value.retryable is True

    @patch("amperstand_core.youtube.anysite.youtube_video")
    @patch("amperstand_core.youtube._oembed", return_value=_OEMBED)
    def test_disabled_without_token(self, mock_oembed, mock_video, monkeypatch):
        monkeypatch.delenv("ANYSITE_ACCESS_TOKEN")
        with pytest.raises(YouTubeTranscriptUnavailable) as exc:
            extract_youtube(URL)
        assert "ANYSITE_ACCESS_TOKEN" in str(exc.value)
        mock_video.assert_not_called()

    @patch("amperstand_core.youtube.anysite.youtube_video", return_value=_VIDEO_WITH_SUBS)
    @patch("amperstand_core.youtube._oembed", return_value={})
    def test_channel_falls_back_to_record_author(self, mock_oembed, mock_video):
        doc = extract_youtube(URL)
        assert doc.author == "Asianometry"
        assert doc.title == "Untitled Video"
