import asyncio
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, JsonCssExtractionStrategy

from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    DEFAULT_KEYWORDS_WORKSHEET,
    DEFAULT_SUMMARY_WORKSHEET,
    write_results_to_google_sheets,
)
from mistral_sentiment_app.llm_analysis import DEFAULT_ANALYSIS_TOPIC, DEFAULT_LLM_PROVIDER, analyze_sentiment
from mistral_sentiment_app.models import CommentRecord, PostRecord

REDDIT_BASE_URL = "https://www.reddit.com"
OLD_REDDIT_BASE_URL = "https://old.reddit.com"
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
    reddit_crawl_concurrency: int = 4
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
    if not permalink:
        return ""
    if permalink.startswith("http://") or permalink.startswith("https://"):
        return permalink
    return f"{REDDIT_BASE_URL}{permalink}"


def _strip_html(value: str) -> str:
    text = re.sub(r"<br\\s*/?>", "\n", value, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_int(value: str | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text:
        return default

    multiplier = 1
    if text.endswith("k"):
        text = text[:-1]
        multiplier = 1_000
    elif text.endswith("m"):
        text = text[:-1]
        multiplier = 1_000_000

    text = text.replace(",", "")
    try:
        return int(float(text) * multiplier)
    except ValueError:
        pass

    match = re.search(r"-?\\d+", text)
    if match:
        return int(match.group(0))
    return default


def _safe_timestamp_seconds(raw_value: str | int | float | None) -> float:
    if raw_value is None:
        return 0.0

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return 0.0
        try:
            value = float(text)
        except ValueError:
            # old.reddit exposes comment timestamps as ISO-8601 datetimes
            normalized = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return 0.0
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
    else:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0

    if value > 1_000_000_000_000:
        return value / 1000.0
    return value


def _as_list_from_extracted_content(extracted_content: str | list | dict | None) -> list[dict]:
    if extracted_content is None:
        return []

    data = extracted_content
    if isinstance(data, str):
        payload = data.strip()
        if not payload:
            return []
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


async def _crawl_extract(
    crawler: AsyncWebCrawler,
    *,
    url: str,
    schema: dict,
) -> list[dict]:
    run_config = CrawlerRunConfig(extraction_strategy=JsonCssExtractionStrategy(schema))
    result = await crawler.arun(url=url, config=run_config)
    if not result.success:
        return []
    return _as_list_from_extracted_content(getattr(result, "extracted_content", None))


async def _fetch_post_listing(
    crawler: AsyncWebCrawler,
    subreddit_name: str,
    *,
    after_fullname: str | None,
    count: int,
) -> list[dict]:
    query_parts = ["limit=100"]
    if after_fullname:
        query_parts.append(f"after={after_fullname}")
        query_parts.append(f"count={count}")
    url = f"{OLD_REDDIT_BASE_URL}/r/{subreddit_name}/new/?{'&'.join(query_parts)}"

    schema = {
        "name": "subreddit_posts",
        "baseSelector": "div.thing.link",
        "fields": [
            {"name": "fullname", "type": "attribute", "attribute": "data-fullname"},
            {"name": "id", "type": "attribute", "attribute": "data-fullname"},
            {"name": "title", "selector": "a.title", "type": "text"},
            {"name": "score", "type": "attribute", "attribute": "data-score"},
            {"name": "timestamp", "type": "attribute", "attribute": "data-timestamp"},
            {"name": "permalink", "type": "attribute", "attribute": "data-permalink"},
        ],
    }
    return await _crawl_extract(crawler, url=url, schema=schema)


async def _fetch_post_page_details(
    crawler: AsyncWebCrawler,
    post_url: str,
) -> tuple[str, list[dict]]:
    result = await crawler.arun(url=post_url, config=CrawlerRunConfig())
    if not result.success:
        return "", []

    html = str(getattr(result, "html", "") or getattr(result, "cleaned_html", "") or "")
    if not html:
        return "", []

    soup = BeautifulSoup(html, "html.parser")

    body_html = ""
    post_node = soup.select_one("div.thing.link")
    if post_node is not None:
        body_node = post_node.select_one("div.usertext-body div.md")
        if body_node is not None:
            body_html = str(body_node)

    comment_rows: list[dict] = []
    for comment_node in soup.select("div.thing.comment"):
        body_node = comment_node.select_one("div.usertext-body div.md")
        permalink_node = comment_node.select_one("a.bylink[data-event-action='permalink']")
        timestamp_node = comment_node.select_one("time.live-timestamp")
        score_node = comment_node.select_one("span.score")
        comment_id = comment_node.get("data-fullname", "") or comment_node.get("id", "")
        if comment_id.startswith("thing_"):
            comment_id = comment_id.removeprefix("thing_")
        comment_rows.append(
            {
                "id": comment_id,
                "score": comment_node.get("data-score", "") or (score_node.get_text(strip=True) if score_node else ""),
                "timestamp": comment_node.get("data-timestamp", "") or (timestamp_node.get("datetime", "") if timestamp_node else ""),
                "permalink": permalink_node.get("href", "") if permalink_node else "",
                "body": str(body_node) if body_node is not None else "",
            }
        )

    return _strip_html(body_html), comment_rows


def _comment_id(value: str) -> str:
    if not value:
        return ""
    return value.removeprefix("t1_").strip()


def _post_id(value: str) -> str:
    if not value:
        return ""
    return value.removeprefix("t3_").strip()


def _sync_fetch_weekly_data_with_crawl4ai(
    subreddit_name: str,
    since_utc: float,
    until_utc: float,
    post_limit: int,
    comment_limit: int,
    crawl_concurrency: int,
    user_agent: str,
) -> tuple[list[PostRecord], list[CommentRecord]]:
    async def runner() -> tuple[list[PostRecord], list[CommentRecord]]:
        browser_config = BrowserConfig(headless=True, user_agent=user_agent)
        semaphore = asyncio.Semaphore(max(1, crawl_concurrency))

        posts: list[PostRecord] = []
        comments: list[CommentRecord] = []
        seen_comment_ids: set[str] = set()
        after_fullname: str | None = None
        count = 0
        listing_page_cap = 30

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for _ in range(listing_page_cap):
                listing_items = await _fetch_post_listing(
                    crawler,
                    subreddit_name=subreddit_name,
                    after_fullname=after_fullname,
                    count=count,
                )
                if not listing_items:
                    break

                oldest_seen = float("inf")
                candidates: list[tuple[dict, float, str, str]] = []
                for item in listing_items:
                    created_utc = _safe_timestamp_seconds(item.get("timestamp"))
                    if created_utc > 0:
                        oldest_seen = min(oldest_seen, created_utc)

                    if created_utc < since_utc or created_utc > until_utc:
                        continue

                    permalink = full_reddit_url(str(item.get("permalink", "") or ""))
                    post_url = permalink or ""
                    if not post_url:
                        continue
                    crawl_post_url = post_url.replace(REDDIT_BASE_URL, OLD_REDDIT_BASE_URL, 1)
                    candidates.append((item, created_utc, permalink, crawl_post_url))

                async def fetch_with_limit(url: str) -> tuple[str, list[dict]]:
                    async with semaphore:
                        return await _fetch_post_page_details(crawler, url)

                details_list = await asyncio.gather(
                    *(fetch_with_limit(crawl_url) for _, _, _, crawl_url in candidates)
                )

                for (item, created_utc, permalink, _), (body_text, comment_rows) in zip(
                    candidates,
                    details_list,
                    strict=True,
                ):
                    if len(posts) < post_limit:
                        post_record = PostRecord(
                            id=_post_id(str(item.get("id", "") or "")),
                            title=str(item.get("title", "") or "").strip(),
                            body=body_text,
                            score=_safe_int(item.get("score")),
                            created_utc=created_utc,
                            permalink=permalink,
                        )
                        posts.append(post_record)

                    if len(posts) >= post_limit and len(comments) >= comment_limit:
                        break

                    if len(comments) < comment_limit:
                        for row in comment_rows:
                            comment_created_utc = _safe_timestamp_seconds(row.get("timestamp"))
                            if comment_created_utc < since_utc or comment_created_utc > until_utc:
                                continue

                            body = _strip_html(str(row.get("body", "") or ""))
                            if not body or body in {"[deleted]", "[removed]"}:
                                continue

                            raw_comment_id = _comment_id(str(row.get("id", "") or ""))
                            parsed = urlparse(str(row.get("permalink", "") or ""))
                            if not raw_comment_id and parsed.fragment:
                                raw_comment_id = parsed.fragment
                            if not raw_comment_id:
                                continue
                            if raw_comment_id in seen_comment_ids:
                                continue
                            seen_comment_ids.add(raw_comment_id)

                            comment_permalink = str(row.get("permalink", "") or "")
                            if comment_permalink and not comment_permalink.startswith("http"):
                                comment_permalink = urljoin(REDDIT_BASE_URL, comment_permalink)
                            if comment_permalink.startswith(OLD_REDDIT_BASE_URL):
                                comment_permalink = comment_permalink.replace(
                                    OLD_REDDIT_BASE_URL,
                                    REDDIT_BASE_URL,
                                    1,
                                )

                            comments.append(
                                CommentRecord(
                                    id=raw_comment_id,
                                    body=body,
                                    score=_safe_int(row.get("score")),
                                    created_utc=comment_created_utc,
                                    permalink=comment_permalink,
                                )
                            )
                            if len(comments) >= comment_limit:
                                break

                if len(posts) >= post_limit and len(comments) >= comment_limit:
                    break

                if oldest_seen < since_utc:
                    break

                last_fullname = str(listing_items[-1].get("fullname", "") or "").strip()
                if not last_fullname:
                    break
                after_fullname = last_fullname
                count += len(listing_items)

        return posts[:post_limit], comments[:comment_limit]

    return asyncio.run(runner())


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

    posts, comments = _sync_fetch_weekly_data_with_crawl4ai(
        subreddit_name=options.subreddit,
        since_utc=since_utc,
        until_utc=until_utc,
        post_limit=options.reddit_post_limit,
        comment_limit=options.reddit_comment_limit,
        crawl_concurrency=options.reddit_crawl_concurrency,
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