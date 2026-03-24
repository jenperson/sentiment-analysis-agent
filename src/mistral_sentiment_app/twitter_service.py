import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    write_results_to_google_sheets,
)
from mistral_sentiment_app.llm_analysis import DEFAULT_ANALYSIS_TOPIC, DEFAULT_LLM_PROVIDER, analyze_sentiment
from mistral_sentiment_app.models import CommentRecord, PostRecord
from mistral_sentiment_app.service import AnalysisOptions, build_result, compute_window, keyword_mentions, load_keywords

TWITTER_API_BASE_URL = "https://api.twitter.com/2"
DEFAULT_TWITTER_QUERY = "MistralAI OR \"Mistral AI\""
DEFAULT_TWITTER_POST_LIMIT = 100
DEFAULT_TWITTER_REPLY_LIMIT = 300
DEFAULT_TWITTER_MAX_CONVERSATIONS = 20


@dataclass
class TwitterAnalysisOptions:
    query: str = DEFAULT_TWITTER_QUERY
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
    twitter_post_limit: int = DEFAULT_TWITTER_POST_LIMIT
    twitter_reply_limit: int = DEFAULT_TWITTER_REPLY_LIMIT
    twitter_max_conversations: int = DEFAULT_TWITTER_MAX_CONVERSATIONS
    write_google_sheets: bool = False
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID
    google_sheets_summary_worksheet: str = "twitter_sentiment_summary"
    google_sheets_keywords_worksheet: str = "twitter_keyword_mentions"


def _get_required_twitter_bearer_token() -> str:
    token = os.getenv("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: TWITTER_BEARER_TOKEN")
    return token


def _to_rfc3339(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_twitter_datetime(value: str) -> float:
    # Example: 2026-03-23T12:34:56.000Z
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()


def _clamp_recent_search_window(start_time: datetime, end_time: datetime) -> tuple[datetime, datetime]:
    # X recent search rejects windows slightly older than 7 days; keep a safety buffer.
    now_utc = datetime.now(timezone.utc)
    latest_allowed_end = now_utc - timedelta(seconds=20)
    clamped_end = end_time if end_time <= latest_allowed_end else latest_allowed_end

    oldest_allowed_start = now_utc - timedelta(days=6, hours=23, minutes=59)
    clamped_start = start_time if start_time >= oldest_allowed_start else oldest_allowed_start

    if clamped_start >= clamped_end:
        clamped_start = clamped_end - timedelta(minutes=1)

    return clamped_start, clamped_end


def _search_recent_tweets(
    *,
    bearer_token: str,
    query: str,
    start_time: datetime,
    end_time: datetime,
    max_results: int,
    next_token: str | None = None,
) -> dict:
    start_time, end_time = _clamp_recent_search_window(start_time, end_time)
    params = {
        "query": query,
        "start_time": _to_rfc3339(start_time),
        "end_time": _to_rfc3339(end_time),
        "max_results": max(10, min(max_results, 100)),
        "tweet.fields": "id,text,author_id,created_at,public_metrics,conversation_id",
        "user.fields": "username,name",
        "expansions": "author_id",
    }
    if next_token:
        params["next_token"] = next_token

    response = requests.get(
        f"{TWITTER_API_BASE_URL}/tweets/search/recent",
        headers={"Authorization": f"Bearer {bearer_token}"},
        params=params,
        timeout=30,
    )
    if not response.ok:
        detail = response.text
        try:
            payload = response.json()
            detail = str(payload)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            "Twitter recent search request failed: "
            f"status={response.status_code}, query={query!r}, "
            f"start_time={params['start_time']}, end_time={params['end_time']}, detail={detail}"
        )
    return response.json()


def _author_lookup(payload: dict) -> dict[str, dict]:
    includes = payload.get("includes", {})
    users = includes.get("users", [])
    return {str(user.get("id")): user for user in users}


def _score_from_public_metrics(metrics: dict | None) -> int:
    if not metrics:
        return 0
    return int(metrics.get("like_count", 0)) + int(metrics.get("retweet_count", 0))


def _convert_tweets_to_posts(tweets: list[dict], users_by_id: dict[str, dict]) -> list[PostRecord]:
    out: list[PostRecord] = []
    for tweet in tweets:
        author = users_by_id.get(str(tweet.get("author_id")), {})
        username = author.get("username", "unknown")
        tweet_id = str(tweet.get("id", ""))
        if not tweet_id:
            continue
        text = str(tweet.get("text", "")).strip()
        if not text:
            continue
        out.append(
            PostRecord(
                id=tweet_id,
                title=f"@{username}",
                body=text,
                score=_score_from_public_metrics(tweet.get("public_metrics")),
                created_utc=_parse_twitter_datetime(str(tweet.get("created_at"))),
                permalink=f"https://twitter.com/{username}/status/{tweet_id}",
            )
        )
    return out


def _fetch_replies_for_conversations(
    *,
    bearer_token: str,
    posts: list[PostRecord],
    start_time: datetime,
    end_time: datetime,
    reply_limit: int,
    max_conversations: int,
) -> list[CommentRecord]:
    replies: list[CommentRecord] = []
    post_ids = [post.id for post in posts[:max_conversations]]

    for post_id in post_ids:
        query = f"conversation_id:{post_id} -is:retweet"
        next_token: str | None = None

        while len(replies) < reply_limit:
            payload = _search_recent_tweets(
                bearer_token=bearer_token,
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=min(100, reply_limit - len(replies)),
                next_token=next_token,
            )

            users_by_id = _author_lookup(payload)
            for tweet in payload.get("data", []):
                tweet_id = str(tweet.get("id", ""))
                if not tweet_id or tweet_id == post_id:
                    continue

                author = users_by_id.get(str(tweet.get("author_id")), {})
                username = author.get("username", "unknown")
                text = str(tweet.get("text", "")).strip()
                if not text:
                    continue

                replies.append(
                    CommentRecord(
                        id=tweet_id,
                        body=text,
                        score=int(tweet.get("public_metrics", {}).get("like_count", 0)),
                        created_utc=_parse_twitter_datetime(str(tweet.get("created_at"))),
                        permalink=f"https://twitter.com/{username}/status/{tweet_id}",
                    )
                )

                if len(replies) >= reply_limit:
                    break

            next_token = payload.get("meta", {}).get("next_token")
            if not next_token:
                break

        if len(replies) >= reply_limit:
            break

    return replies


def fetch_twitter_data(options: TwitterAnalysisOptions, window_start: datetime, window_end: datetime) -> tuple[list[PostRecord], list[CommentRecord]]:
    bearer_token = _get_required_twitter_bearer_token()

    posts: list[PostRecord] = []
    next_token: str | None = None

    while len(posts) < options.twitter_post_limit:
        payload = _search_recent_tweets(
            bearer_token=bearer_token,
            query=options.query,
            start_time=window_start,
            end_time=window_end,
            max_results=min(100, options.twitter_post_limit - len(posts)),
            next_token=next_token,
        )

        users_by_id = _author_lookup(payload)
        page_posts = _convert_tweets_to_posts(payload.get("data", []), users_by_id)
        posts.extend(page_posts)

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break

    comments = _fetch_replies_for_conversations(
        bearer_token=bearer_token,
        posts=posts,
        start_time=window_start,
        end_time=window_end,
        reply_limit=options.twitter_reply_limit,
        max_conversations=options.twitter_max_conversations,
    )

    return posts[: options.twitter_post_limit], comments[: options.twitter_reply_limit]


def run_twitter_analysis(options: TwitterAnalysisOptions) -> dict:
    keywords = load_keywords(Path(options.keywords_file))

    window_start, window_end, window_label = compute_window(
        AnalysisOptions(
            days=options.days,
            start_days_ago=options.start_days_ago,
            end_days_ago=options.end_days_ago,
            date=options.date,
            start_date=options.start_date,
            end_date=options.end_date,
        )
    )

    posts, comments = fetch_twitter_data(options, window_start, window_end)
    mentions = keyword_mentions(keywords=keywords, posts=posts, comments=comments)

    if not posts and not comments:
        sentiment = {
            "average_sentiment": None,
            "summary": "No tweets or replies were found in the selected window.",
            "method_notes": "LLM analysis skipped because the Twitter API returned an empty dataset.",
        }
        result = build_result(
            subreddit_name=f"Twitter query: {options.query}",
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            posts=posts,
            comments=comments,
            sentiment=sentiment,
            mentions=mentions,
            analysis_provider=options.provider,
            analysis_model=options.model_override or "",
            top_posts_method="upvotes",
        )
        result["query"] = options.query
        result["source"] = "twitter"
        return result

    sentiment, analysis_provider, analysis_model = analyze_sentiment(
        posts=posts,
        comments=comments,
        subreddit_name=f"Twitter query: {options.query}",
        window_label=window_label,
        provider=options.provider,
        model_override=options.model_override,
        topic=options.topic,
    )

    result = build_result(
        subreddit_name=f"Twitter query: {options.query}",
        window_start=window_start,
        window_end=window_end,
        window_label=window_label,
        posts=posts,
        comments=comments,
        sentiment=sentiment,
        mentions=mentions,
        analysis_provider=analysis_provider,
        analysis_model=analysis_model,
        top_posts_method="upvotes",
    )
    result["query"] = options.query
    result["source"] = "twitter"

    if options.write_google_sheets:
        result["google_sheets_export"] = write_results_to_google_sheets(
            result=result,
            spreadsheet_id=options.google_sheets_spreadsheet_id,
            summary_worksheet_name=options.google_sheets_summary_worksheet,
            keywords_worksheet_name=options.google_sheets_keywords_worksheet,
        )

    return result
