import argparse
import json
import os
import re
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    DEFAULT_KEYWORDS_WORKSHEET,
    DEFAULT_SUMMARY_WORKSHEET,
    write_results_to_google_sheets,
)
from mistral_sentiment_app.llm_analysis import DEFAULT_LLM_PROVIDER, analyze_sentiment
from mistral_sentiment_app.models import CommentRecord, PostRecord

REDDIT_BASE_URL = "https://www.reddit.com"
DEFAULT_SUBREDDIT = "MistralAI"
DEFAULT_PUBLIC_REDDIT_USER_AGENT = "mistral-weekly-sentiment/0.1"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


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
        "--date",
        default=None,
        help="Specific UTC date to analyze in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive UTC start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive UTC end date in YYYY-MM-DD format. Defaults to --start-date when omitted.",
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
    parser.add_argument(
        "--provider",
        choices=["mistral", "claude"],
        default=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
        help="LLM provider for sentiment analysis (default: mistral)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", ""),
        help="Optional model override for the selected provider",
    )
    parser.add_argument(
        "--write-google-sheets",
        action="store_true",
        default=env_flag("GOOGLE_SHEETS_WRITE", default=False),
        help="Append the run results to Google Sheets",
    )
    parser.add_argument(
        "--google-sheets-spreadsheet-id",
        default=os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID),
        help="Spreadsheet ID for Google Sheets export",
    )
    parser.add_argument(
        "--google-sheets-summary-worksheet",
        default=os.getenv("GOOGLE_SHEETS_SUMMARY_WORKSHEET", DEFAULT_SUMMARY_WORKSHEET),
        help="Worksheet name for summary and sentiment data",
    )
    parser.add_argument(
        "--google-sheets-keywords-worksheet",
        default=os.getenv("GOOGLE_SHEETS_KEYWORDS_WORKSHEET", DEFAULT_KEYWORDS_WORKSHEET),
        help="Worksheet name for keyword mention data",
    )
    return parser.parse_args()


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


def parse_utc_date(date_text: str, argument_name: str) -> datetime:
    try:
        parsed = datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError as exc:
        raise RuntimeError(f"{argument_name} must be in YYYY-MM-DD format") from exc
    return parsed.replace(tzinfo=timezone.utc)


def compute_window(args: argparse.Namespace) -> tuple[datetime, datetime, str]:
    now = datetime.now(timezone.utc)

    has_day_range = args.start_days_ago is not None or args.end_days_ago is not None
    has_date_range = any(value is not None for value in (args.date, args.start_date, args.end_date))

    if args.date is not None and (args.start_date is not None or args.end_date is not None):
        raise RuntimeError("Use either --date or --start-date/--end-date, not both")

    if has_date_range and has_day_range:
        raise RuntimeError(
            "Choose either day-based filters (--start-days-ago/--end-days-ago) or date-based filters (--date/--start-date/--end-date)"
        )

    if args.date is not None:
        start = parse_utc_date(args.date, "--date")
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return start, end, f"on {args.date}"

    if args.start_date is not None or args.end_date is not None:
        if args.start_date is None:
            raise RuntimeError("--start-date is required when --end-date is provided")

        start = parse_utc_date(args.start_date, "--start-date")
        end_date = args.start_date if args.end_date is None else args.end_date
        end = parse_utc_date(end_date, "--end-date") + timedelta(days=1) - timedelta(microseconds=1)
        if start > end:
            raise RuntimeError("--start-date must be on or before --end-date")
        return start, end, f"from {args.start_date} to {end_date}"

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
    sentiment, analysis_provider, analysis_model = analyze_sentiment(
        posts=posts,
        comments=comments,
        subreddit_name=args.subreddit,
        window_label=window_label,
        provider=args.provider,
        model_override=args.model,
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
        analysis_provider=analysis_provider,
        analysis_model=analysis_model,
    )

    if args.write_google_sheets:
        result["google_sheets_export"] = write_results_to_google_sheets(
            result=result,
            spreadsheet_id=args.google_sheets_spreadsheet_id,
            summary_worksheet_name=args.google_sheets_summary_worksheet,
            keywords_worksheet_name=args.google_sheets_keywords_worksheet,
        )

    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
