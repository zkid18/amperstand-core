"""Reddit post capture via Anysite.

old.reddit.com now serves a login wall to anything that isn't a signed-in
browser, so neither a direct fetch nor a rendered one gets the post. Anysite's
Reddit endpoint returns it structured, which is also cheaper to turn into a
doc than scraping the page would be.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from amperstand_core import anysite
from amperstand_core.models import CapturedContent, ContentType

_REDDIT_POST_RE = re.compile(
    r"://(?:www\.|old\.|new\.)?reddit\.com/r/[^/]+/comments/[^/?#]+", re.IGNORECASE,
)


def is_reddit_url(url: str) -> bool:
    return bool(_REDDIT_POST_RE.search(url))


def extract_reddit(url: str) -> CapturedContent:
    post = anysite.reddit_post(url)

    title = (post.get("title") or "").strip() or "Reddit post"
    author = (post.get("author") or {}).get("name")
    subreddit = post.get("subreddit")
    text = (post.get("text") or "").strip()
    link = post.get("content_url")

    lines: list[str] = []
    if subreddit:
        lines.append(f"**Subreddit**: {subreddit}")
    if author:
        lines.append(f"**Author**: u/{author}")
    stats = []
    if post.get("vote_count") is not None:
        stats.append(f"{post['vote_count']} points")
    if post.get("comment_count") is not None:
        stats.append(f"{post['comment_count']} comments")
    if stats:
        lines.append(f"**Score**: {', '.join(stats)}")
    created = post.get("created_at")
    if isinstance(created, (int, float)):
        posted = datetime.fromtimestamp(created, tz=timezone.utc)
        lines.append(f"**Posted**: {posted.strftime('%Y-%m-%d')}")
    if lines:
        lines.append("")

    if text:
        lines.append(text)
    elif link:
        # A link post has no body of its own; the link is the content.
        lines.append(f"[{link}]({link})")
    else:
        lines.append("*No text in this post.*")

    return CapturedContent(
        url=url,
        title=title,
        content_markdown="\n".join(lines),
        content_type=ContentType.ARTICLE,
        author=author,
    )
