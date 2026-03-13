import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    DEFAULT_KEYWORDS_WORKSHEET,
    DEFAULT_SUMMARY_WORKSHEET,
    write_results_to_google_sheets,
)
from mistral_sentiment_app.llm_analysis import DEFAULT_ANALYSIS_TOPIC, DEFAULT_LLM_PROVIDER, analyze_sentiment
from mistral_sentiment_app.models import CommentRecord, PostRecord

REDDIT_BASE_URL = "https://www.reddit.com"
DEFAULT_SUBREDDIT = "MistralAI"
DEFAULT_PUBLIC_REDDIT_USER_AGENT = "mistral-weekly-sentiment/0.1"


@dataclass
class AnalysisOptions:
    subreddit: str = DEFAULT_SUBREDDIT
    days: int = 7
    start_days_ago: int | None = None
    end_days_ago: int | None = None
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    keywords_file: str = "keywords.txt"
    topic: str = DEFAULT_ANALYSIS_TOPIC
    provider: str = DEFAULT_LLM_PROVIDER
    model_override: str = ""
    reddit_post_limit: int = 300
    reddit_comment_limit: int = 1500
    reddit_user_agent: str = DEFAULT_PUBLIC_REDDIT_USER_AGENT
    write_google_sheets: bool = False
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID
    google_sheets_summary_worksheet: str = DEFAULT_SUMMARY_WORKSHEET
    google_sheets_keywords_worksheet: str = DEFAULT_KEYWORDS_WORKSHEET


def parse_utc_date(date_text: str, argument_name: str) -> datetime:
    try:
        parsed = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as exc:
        raise RuntimeError(f"{argument_name} must be in YYYY-MM-DD format") from exc
    return parsed.replace(tzinfo=timezone.utc)


def compute_window(options: AnalysisOptions) -> tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)

    has_day_range = options.start_days_ago is not None or options.end_days_ago is not None
    has_date_range = any(
        value is not None for value in (options.date, options.start_date, options.end_date)
    )

    if options.date is not None and (options.start_date is not None or options.end_date is not None):
        raise RuntimeError("Use either --date or --start-date/--end-date, not both")

    if has_date_range and has_day_range:
        raise RuntimeError(
            "Choose either day-based filters (--start-days-ago/--end-days-ago) or date-based filters (--date/--start-date/--end-date)"
        )

    if options.date is not None:
        start = parse_utc_date(options.date, "--date")
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return start, end, f"on {options.date}"

    if options.start_date is not None or options.end_date is not None:
        if options.start_date is None:
            raise RuntimeError("--start-date is required when --end-date is provided")

        start = parse_utc_date(options.start_date, "--start-date")
        end_date = options.start_date if options.end_date is None else options.end_date
        end = parse_utc_date(end_date, "--end-date") + timedelta(days=1) - timedelta(
            microseconds=1
        )
        if start > end:
            raise RuntimeError("--start-date must be on or before --end-date")
        return start, end, f"from {options.start_date} to {end_date}"

    if options.start_days_ago is None and options.end_days_ago is None:
        if options.days < 1:
            raise RuntimeError("--days must be >= 1")
        start = now - timedelta(days=options.days)
        end = now
        return start, end, f"last {options.days} days"

    if options.start_days_ago is None:
        raise RuntimeError("--start-days-ago is required when --end-days-ago is provided")

    start_days_ago = options.start_days_ago
    end_days_ago = 0 if options.end_days_ago is None else options.end_days_ago

    if start_days_ago < 0 or end_days_ago < 0:
        raise RuntimeError("--start-days-ago and --end-days-ago must be >= 0")
    if start_days_ago < end_days_ago:
        raise RuntimeError("--start-days-ago must be >= --end-days-ago")

    start = now - timedelta(days=start_days_ago)
    end = now - timedelta(days=end_days_ago)
    return start, end, f"from {start_days_ago} to {end_days_ago} days ago"


def _parse_keyword_lines(lines: Iterable[str]) -> list[str]:
    return [
        kw for raw in lines if (kw := raw.strip()) and not kw.startswith("#")
    ]


def load_keywords(path: Path) -> list[str]:
    env_value = os.getenv("KEYWORDS", "").strip()
    if env_value:
        keywords = _parse_keyword_lines(env_value.splitlines())
        if keywords:
            return keywords

    if not path.exists():
        raise FileNotFoundError(f"Keyword file not found: {path}")
    keywords = _parse_keyword_lines(path.read_text(encoding="utf-8").splitlines())
    if not keywords:
        raise RuntimeError("No keywords found in keywords file.")
    return keywords


def full_reddit_url(permalink: str) -> str:
    return f"{REDDIT_BASE_URL}{permalink}"


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
    sorted_posts = sorted(posts, key=lambda item: item.score, reverse=True)[:n]
    out: list[dict] = []
    for post in sorted_posts:
        content = post.title.strip()
        if post.body.strip():
            content = f"{content}\n\n{post.body.strip()}"
        out.append(
            {
                "post_link": post.permalink,
                "content": content,
                "upvotes": post.score,
            }
        )
    return out


def keyword_mentions(
    keywords: list[str],
    posts: Iterable[PostRecord],
    comments: Iterable[CommentRecord],
) -> dict[str, dict]:
    matches: dict[str, dict] = {keyword: {"count": 0, "links": []} for keyword in keywords}
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


def build_result(
    subreddit_name: str,
    window_start: datetime,
    window_end: datetime,
    window_label: str,
    posts: list[PostRecord],
    comments: list[CommentRecord],
    sentiment: dict,
    mentions: dict[str, dict],
    analysis_provider: str,
    analysis_model: str,
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
        "analysis": {
            "provider": analysis_provider,
            "model": analysis_model,
        },
        "average_sentiment": sentiment["average_sentiment"],
        "summary_of_week": sentiment["summary"],
        "sentiment_method_notes": sentiment.get("method_notes", ""),
        "top_3_posts_by_upvotes": top_posts_by_upvotes(posts, n=3),
        "keyword_mentions": mentions,
    }


def run_analysis(options: AnalysisOptions) -> dict:
    keywords = load_keywords(Path(options.keywords_file))
    window_start, window_end, window_label = compute_window(options)
    since_utc = window_start.timestamp()
    until_utc = window_end.timestamp()

    posts = fetch_weekly_posts_public(
        subreddit_name=options.subreddit,
        since_utc=since_utc,
        until_utc=until_utc,
        post_limit=options.reddit_post_limit,
        user_agent=options.reddit_user_agent,
    )
    comments = fetch_weekly_comments_public(
        subreddit_name=options.subreddit,
        since_utc=since_utc,
        until_utc=until_utc,
        comment_limit=options.reddit_comment_limit,
        user_agent=options.reddit_user_agent,
    )

    mentions = keyword_mentions(keywords=keywords, posts=posts, comments=comments)
    sentiment, analysis_provider, analysis_model = analyze_sentiment(
        posts=posts,
        comments=comments,
        subreddit_name=options.subreddit,
        window_label=window_label,
        provider=options.provider,
        model_override=options.model_override,
        topic=options.topic,
    )
    result = build_result(
        subreddit_name=options.subreddit,
        window_start=window_start,
        window_end=window_end,
        window_label=window_label,
        posts=posts,
        comments=comments,
        sentiment=sentiment,
        mentions=mentions,
        analysis_provider=analysis_provider,
        analysis_model=analysis_model,
    )

    if options.write_google_sheets:
        result["google_sheets_export"] = write_results_to_google_sheets(
            result=result,
            spreadsheet_id=options.google_sheets_spreadsheet_id,
            summary_worksheet_name=options.google_sheets_summary_worksheet,
            keywords_worksheet_name=options.google_sheets_keywords_worksheet,
        )

    return result