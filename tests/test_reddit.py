"""Reddit capture via Anysite."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from amperstand_core.models import ContentType
from amperstand_core.reddit import extract_reddit, is_reddit_url


class TestIsRedditUrl:
    @pytest.mark.parametrize("url", [
        "https://old.reddit.com/r/MachineLearning/comments/62rd9a/",
        "https://www.reddit.com/r/python/comments/abc123/some_title/",
        "https://reddit.com/r/x/comments/1o2g2pq/i_think_i_need_to_rehome_my_dog/",
        "https://new.reddit.com/r/x/comments/zz9/",
    ])
    def test_post_urls(self, url: str):
        assert is_reddit_url(url) is True

    @pytest.mark.parametrize("url", [
        "https://www.reddit.com/r/python/",
        "https://www.reddit.com/user/someone/",
        "https://example.com/r/x/comments/1/",
        "https://www.linkedin.com/posts/x",
    ])
    def test_non_post_urls(self, url: str):
        assert is_reddit_url(url) is False


_LINK_POST = {
    "id": "62rd9a",
    "title": "[R] OpenAI awarded $30 million from the Open Philanthropy Project",
    "author": {"name": "downtownslim"},
    "subreddit": "r/MachineLearning",
    "created_at": 1491032245,
    "vote_count": 116,
    "comment_count": 49,
    "post_type": "link",
    "content_url": "http://www.openphilanthropy.org/focus/openai-general-support",
    "text": None,
}


class TestExtractReddit:
    @patch("amperstand_core.reddit.anysite.reddit_post", return_value=_LINK_POST)
    def test_link_post_keeps_the_link_as_content(self, mock_post):
        url = "https://old.reddit.com/r/MachineLearning/comments/62rd9a/"
        doc = extract_reddit(url)

        assert doc.url == url
        assert doc.content_type == ContentType.ARTICLE
        assert doc.title.startswith("[R] OpenAI awarded")
        assert doc.author == "downtownslim"
        md = doc.content_markdown
        assert "**Subreddit**: r/MachineLearning" in md
        assert "**Author**: u/downtownslim" in md
        assert "116 points, 49 comments" in md
        assert "**Posted**: 2017-04-01" in md
        assert "openphilanthropy.org" in md
        mock_post.assert_called_once_with(url)

    @patch("amperstand_core.reddit.anysite.reddit_post")
    def test_text_post_body(self, mock_post):
        mock_post.return_value = {
            **_LINK_POST, "post_type": "self", "content_url": None,
            "text": "First paragraph.\n\nSecond paragraph.",
        }
        doc = extract_reddit("https://www.reddit.com/r/x/comments/1/")
        assert "Second paragraph." in doc.content_markdown
        assert "openphilanthropy" not in doc.content_markdown

    @patch("amperstand_core.reddit.anysite.reddit_post")
    def test_sparse_record_does_not_crash(self, mock_post):
        mock_post.return_value = {"id": "x"}
        doc = extract_reddit("https://www.reddit.com/r/x/comments/x/")
        assert doc.title == "Reddit post"
        assert "*No text in this post.*" in doc.content_markdown
