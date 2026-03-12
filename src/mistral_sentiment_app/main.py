import argparse
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

REDDIT_BASE_URL = "https://www.reddit.com"
DEFAULT_SUBREDDIT = "MistralAI"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_PUBLIC_REDDIT_USER_AGENT = "mistral-weekly-sentiment/0.1"


@dataclass
class PostRecord:
    id: str
    title: str
    body: str
    score: int
    created_utc: float
    permalink: str


@dataclass
class CommentRecord:
    id: str
    body: str
    score: int
    created_utc: float
    permalink: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze weekly subreddit sentiment for MistralAI and products."
    )
    parser.add_argument(
        "--subreddit",
        default=DEFAULT_SUBREDDIT,
        help=f"Subreddit to analyze (default: {DEFAULT_SUBREDDIT})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of trailing days to analyze when no explicit range is provided (default: 7)",
    )
    parser.add_argument(
        "--start-days-ago",
        type=int,
        default=None,
        help=(
            "Optional range start in days ago (older boundary). "
            "Example: 14 with --end-days-ago 7 analyzes content from 14 to 7 days ago."
        ),
    )
    parser.add_argument(
        "--end-days-ago",
        type=int,
        default=None,
        help=(
            "Optional range end in days ago (newer boundary). "
            "Defaults to 0 when --start-days-ago is set."
        ),
    )
    parser.add_argument(
        "--keywords-file",
        default="keywords.txt",
        help="Path to newline-delimited keyword file",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path. If omitted, prints to stdout only.",
    )
    return parser.parse_args()


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_keywords(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")
    keywords: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        keyword = raw.strip()
        if keyword and not keyword.startswith("#"):
            keywords.append(keyword)
    if not keywords:
        raise RuntimeError("No keywords found in keywords file.")
    return keywords


def full_reddit_url(permalink: str) -> str:
    return f"{REDDIT_BASE_URL}{permalink}"


def compute_window(args: argparse.Namespace) -> tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)

    if args.start_days_ago is None and args.end_days_ago is None:
        if args.days < 1:
            raise RuntimeError("--days must be >= 1")
        start = now - timedelta(days=args.days)
        end = now
        return start, end, f"last {args.days} days"

    if args.start_days_ago is None:
        raise RuntimeError("--start-days-ago is required when --end-days-ago is provided")

    start_days_ago = args.start_days_ago
    end_days_ago = 0 if args.end_days_ago is None else args.end_days_ago

    if start_days_ago < 0 or end_days_ago < 0:
        raise RuntimeError("--start-days-ago and --end-days-ago must be >= 0")
    if start_days_ago < end_days_ago:
        raise RuntimeError("--start-days-ago must be >= --end-days-ago")

    start = now - timedelta(days=start_days_ago)
    end = now - timedelta(days=end_days_ago)
    return start, end, f"from {start_days_ago} to {end_days_ago} days ago"


def _public_reddit_get(path: str, params: dict, user_agent: str) -> dict:
    url = f"https://www.reddit.com{path}"
    response = requests.get(
        url,
        params={**params, "raw_json": 1},
        headers={"User-Agent": user_agent},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_weekly_posts_public(
    subreddit_name: str,
    since_utc: float,
    until_utc: float,
    post_limit: int,
    user_agent: str,
) -> list[PostRecord]:
    records: list[PostRecord] = []
    after: str | None = None
    page_limit = 100

    while len(records) < post_limit:
        remaining = post_limit - len(records)
        listing = _public_reddit_get(
            path=f"/r/{subreddit_name}/new.json",
            params={"limit": min(page_limit, remaining), "after": after},
            user_agent=user_agent,
        )
        data = listing.get("data", {})
        children = data.get("children", [])
        if not children:
            break

        oldest_created = float("inf")
        for child in children:
            item = child.get("data", {})
            created_utc = float(item.get("created_utc", 0.0))
            oldest_created = min(oldest_created, created_utc)

            if created_utc < since_utc or created_utc > until_utc:
                continue

            records.append(
                PostRecord(
                    id=item.get("id", ""),
                    title=item.get("title", "") or "",
                    body=item.get("selftext", "") or "",
                    score=int(item.get("score", 0) or 0),
                    created_utc=created_utc,
                    permalink=full_reddit_url(item.get("permalink", "")),
                )
            )

        if len(records) >= post_limit:
            break

        after = data.get("after")
        if not after:
            break

        # Listings are newest to oldest. Stop after we pass the older boundary.
        if oldest_created < since_utc:
            break

    return records


def fetch_weekly_comments_public(
    subreddit_name: str,
    since_utc: float,
    until_utc: float,
    comment_limit: int,
    user_agent: str,
) -> list[CommentRecord]:
    records: list[CommentRecord] = []
    after: str | None = None
    page_limit = 100

    while len(records) < comment_limit:
        remaining = comment_limit - len(records)
        listing = _public_reddit_get(
            path=f"/r/{subreddit_name}/comments.json",
            params={"limit": min(page_limit, remaining), "after": after},
            user_agent=user_agent,
        )
        data = listing.get("data", {})
        children = data.get("children", [])
        if not children:
            break

        oldest_created = float("inf")
        for child in children:
            item = child.get("data", {})
            created_utc = float(item.get("created_utc", 0.0))
            oldest_created = min(oldest_created, created_utc)

            if created_utc < since_utc or created_utc > until_utc:
                continue

            body = (item.get("body", "") or "").strip()
            if not body or body in {"[deleted]", "[removed]"}:
                continue

            records.append(
                CommentRecord(
                    id=item.get("id", ""),
                    body=body,
                    score=int(item.get("score", 0) or 0),
                    created_utc=created_utc,
                    permalink=full_reddit_url(item.get("permalink", "")),
                )
            )

        if len(records) >= comment_limit:
            break

        after = data.get("after")
        if not after:
            break

        if oldest_created < since_utc:
            break

    return records


def top_posts_by_upvotes(posts: list[PostRecord], n: int = 3) -> list[dict]:
    sorted_posts = sorted(posts, key=lambda p: p.score, reverse=True)[:n]
    out: list[dict] = []
    for p in sorted_posts:
        content = p.title.strip()
        if p.body.strip():
            content = f"{content}\n\n{p.body.strip()}"
        out.append(
            {
                "post_link": p.permalink,
                "content": content,
                "upvotes": p.score,
            }
        )
    return out


def keyword_mentions(
    keywords: list[str],
    posts: Iterable[PostRecord],
    comments: Iterable[CommentRecord],
) -> dict[str, dict]:
    matches: dict[str, dict] = {
        keyword: {"count": 0, "links": []} for keyword in keywords
    }

    compiled = {
        keyword: re.compile(rf"(?<!\\w){re.escape(keyword)}(?!\\w)", re.IGNORECASE)
        for keyword in keywords
    }

    def process_text(text: str, link: str) -> None:
        for keyword, pattern in compiled.items():
            found = pattern.findall(text)
            if not found:
                continue
            matches[keyword]["count"] += len(found)
            if link not in matches[keyword]["links"]:
                matches[keyword]["links"].append(link)

    for post in posts:
        process_text(f"{post.title}\n{post.body}", post.permalink)

    for comment in comments:
        process_text(comment.body, comment.permalink)

    return matches


def serialize_for_claude(posts: list[PostRecord], comments: list[CommentRecord]) -> dict:
    # Limit payload size to stay within practical context windows while preserving variety.
    max_items = int(os.getenv("CLAUDE_MAX_ITEMS", "250"))
    max_chars = int(os.getenv("CLAUDE_MAX_CHARS_PER_ITEM", "1200"))

    serialized_posts = []
    for p in sorted(posts, key=lambda x: x.score, reverse=True)[:max_items]:
        serialized_posts.append(
            {
                "type": "post",
                "id": p.id,
                "score": p.score,
                "link": p.permalink,
                "text": (f"Title: {p.title}\nBody: {p.body}")[:max_chars],
            }
        )

    serialized_comments = []
    for c in sorted(comments, key=lambda x: x.score, reverse=True)[:max_items]:
        serialized_comments.append(
            {
                "type": "comment",
                "id": c.id,
                "score": c.score,
                "link": c.permalink,
                "text": c.body[:max_chars],
            }
        )

    return {
        "posts": serialized_posts,
        "comments": serialized_comments,
    }


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("Claude response did not contain JSON.")
    return json.loads(match.group(0))


def analyze_with_claude(
    posts: list[PostRecord],
    comments: list[CommentRecord],
    subreddit_name: str,
    window_label: str,
) -> dict:
    api_key = get_required_env("ANTHROPIC_API_KEY")
    model = os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    client = Anthropic(api_key=api_key)

    sentiment_scale = {
        "1": "Very negative, someone is beyond frustrated and unhappy with the brand or products",
        "2": "Somewhat negative, someone is not having a good experience and is expressing frustration, but not hatred",
        "3": "Neutral, someone has a mix of positive and negative things to say leading to a balance, or does not express strong feelings one way or another",
        "4": "Somewhat positive, someone expresses satisfaction with the product or uses it successfully",
        "5": "Very positive, someone is praising the product, and may want to convince others to use it",
    }

    dataset = serialize_for_claude(posts, comments)

    system_prompt = (
        "You are a precise sentiment analyst. Return strict JSON only with no markdown."
    )

    user_prompt = {
        "task": "Analyze sentiment around MistralAI and its products from subreddit data.",
        "scope": {
            "subreddit": subreddit_name,
            "window": window_label,
        },
        "sentiment_scale": sentiment_scale,
        "requirements": {
            "average_sentiment": "Number from 1 to 5, can include decimals.",
            "summary": "Brief summary of major themes this week (3-8 sentences).",
            "method_notes": "Very short explanation of how the score was determined.",
        },
        "output_schema": {
            "average_sentiment": "float",
            "summary": "string",
            "method_notes": "string",
        },
        "data": dataset,
    }

    response = client.messages.create(
        model=model,
        max_tokens=1800,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": json.dumps(user_prompt),
            }
        ],
    )

    text_chunks = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text_chunks.append(block.text)

    raw_text = "\n".join(text_chunks).strip()
    parsed = extract_json_object(raw_text)

    if "average_sentiment" not in parsed or "summary" not in parsed:
        raise RuntimeError("Claude response missing required fields.")

    return parsed


def build_result(
    subreddit_name: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    posts: list[PostRecord],
    comments: list[CommentRecord],
    sentiment: dict,
    mentions: dict[str, dict],
) -> dict:
    duration_days = (window_end - window_start).total_seconds() / 86400

    return {
        "subreddit": subreddit_name,
        "window": {
            "label": window_label,
            "days": round(duration_days, 3),
            "start_utc": window_start.isoformat(),
            "end_utc": window_end.isoformat(),
        },
        "counts": {
            "posts": len(posts),
            "comments": len(comments),
        },
        "average_sentiment": sentiment["average_sentiment"],
        "summary_of_week": sentiment["summary"],
        "sentiment_method_notes": sentiment.get("method_notes", ""),
        "top_3_posts_by_upvotes": top_posts_by_upvotes(posts, n=3),
        "keyword_mentions": mentions,
    }


def main() -> None:
    load_dotenv()
    args = parse_args()

    reddit_post_limit = int(os.getenv("REDDIT_POST_LIMIT", "300"))
    reddit_comment_limit = int(os.getenv("REDDIT_COMMENT_LIMIT", "1500"))

    keywords = load_keywords(Path(args.keywords_file))
    window_start, window_end, window_label = compute_window(args)
    since_utc = window_start.timestamp()
    until_utc = window_end.timestamp()

    public_user_agent = (
        os.getenv("REDDIT_USER_AGENT", "").strip() or DEFAULT_PUBLIC_REDDIT_USER_AGENT
    )
    posts = fetch_weekly_posts_public(
        subreddit_name=args.subreddit,
        since_utc=since_utc,
        until_utc=until_utc,
        post_limit=reddit_post_limit,
        user_agent=public_user_agent,
    )
    comments = fetch_weekly_comments_public(
        subreddit_name=args.subreddit,
        since_utc=since_utc,
        until_utc=until_utc,
        comment_limit=reddit_comment_limit,
        user_agent=public_user_agent,
    )

    mentions = keyword_mentions(keywords=keywords, posts=posts, comments=comments)
    sentiment = analyze_with_claude(
        posts=posts,
        comments=comments,
        subreddit_name=args.subreddit,
        window_label=window_label,
    )

    result = build_result(
        subreddit_name=args.subreddit,
        window_start=window_start,
        window_end=window_end,
        window_label=window_label,
        posts=posts,
        comments=comments,
        sentiment=sentiment,
        mentions=mentions,
    )

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
