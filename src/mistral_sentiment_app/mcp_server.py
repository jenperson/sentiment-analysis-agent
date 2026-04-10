import os

import uvicorn
from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mistral_sentiment_app.discord_service import run_discord_analysis
from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    DEFAULT_KEYWORDS_WORKSHEET,
    DEFAULT_SUMMARY_WORKSHEET,
)
from mistral_sentiment_app.llm_analysis import DEFAULT_ANALYSIS_TOPIC, DEFAULT_LLM_PROVIDER
from mistral_sentiment_app.service import (
    AnalysisOptions,
    DEFAULT_PUBLIC_REDDIT_USER_AGENT,
    DEFAULT_SUBREDDIT,
    run_analysis,
)
from mistral_sentiment_app.twitter_service import (
    DEFAULT_TWITTER_QUERY,
    TwitterAnalysisOptions,
    run_twitter_analysis,
)

mcp = FastMCP(
    name="Mistral Reddit Sentiment",
    instructions=(
        "Analyze sentiment in r/MistralAI posts and comments over a requested time window. "
        "Return structured JSON with sentiment, summary, top posts, and keyword mentions."
    ),
)


@mcp.tool
def analyze_mistral_subreddit(
    subreddit: str = DEFAULT_SUBREDDIT,
    days: int = 7,
    start_days_ago: int | None = None,
    end_days_ago: int | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    keywords_file: str = "keywords.txt",
    topic: str = DEFAULT_ANALYSIS_TOPIC,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = "",
    write_google_sheets: bool = False,
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    google_sheets_summary_worksheet: str = DEFAULT_SUMMARY_WORKSHEET,
    google_sheets_keywords_worksheet: str = DEFAULT_KEYWORDS_WORKSHEET,
    slack_webhook_url: str = "",
) -> dict:
    options = AnalysisOptions(
        subreddit=subreddit,
        days=days,
        start_days_ago=start_days_ago,
        end_days_ago=end_days_ago,
        date=date,
        start_date=start_date,
        end_date=end_date,
        keywords_file=keywords_file,
        topic=topic,
        provider=provider,
        model_override=model,
        reddit_post_limit=int(os.getenv("REDDIT_POST_LIMIT", "300")),
        reddit_comment_limit=int(os.getenv("REDDIT_COMMENT_LIMIT", "1500")),
        reddit_crawl_concurrency=int(os.getenv("REDDIT_CRAWL_CONCURRENCY", "4")),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "").strip() or DEFAULT_PUBLIC_REDDIT_USER_AGENT,
        write_google_sheets=write_google_sheets,
        google_sheets_spreadsheet_id=google_sheets_spreadsheet_id,
        google_sheets_summary_worksheet=google_sheets_summary_worksheet,
        google_sheets_keywords_worksheet=google_sheets_keywords_worksheet,
        slack_webhook_url=slack_webhook_url,
    )
    return run_analysis(options)


@mcp.tool
def analyze_discord_server(
    guild_id: int,
    channel_ids: list[int] | None = None,
    days: int = 7,
    start_days_ago: int | None = None,
    end_days_ago: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    keywords_file: str = "keywords.txt",
    topic: str = DEFAULT_ANALYSIS_TOPIC,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = "",
    write_google_sheets: bool = False,
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    google_sheets_summary_worksheet: str = "discord_sentiment_summary",
    google_sheets_keywords_worksheet: str = "discord_keyword_mentions",
) -> dict:
    """Analyze sentiment in a Discord server's messages over a time period."""
    return run_discord_analysis(
        guild_id=guild_id,
        channel_ids=channel_ids,
        days=days,
        start_days_ago=start_days_ago,
        end_days_ago=end_days_ago,
        start_date=start_date,
        end_date=end_date,
        topic=topic,
        provider=provider,
        model_override=model,
        keywords_file=keywords_file,
        write_google_sheets=write_google_sheets,
        google_sheets_spreadsheet_id=google_sheets_spreadsheet_id,
        google_sheets_summary_worksheet=google_sheets_summary_worksheet,
        google_sheets_keywords_worksheet=google_sheets_keywords_worksheet,
    )


@mcp.tool
def analyze_twitter_query(
    query: str = DEFAULT_TWITTER_QUERY,
    days: int = 7,
    start_days_ago: int | None = None,
    end_days_ago: int | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    keywords_file: str = "keywords.txt",
    topic: str = DEFAULT_ANALYSIS_TOPIC,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = "",
    write_google_sheets: bool = False,
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    google_sheets_summary_worksheet: str = "twitter_sentiment_summary",
    google_sheets_keywords_worksheet: str = "twitter_keyword_mentions",
) -> dict:
    """Analyze sentiment in tweets matching a query over a time period."""
    return run_twitter_analysis(
        TwitterAnalysisOptions(
            query=query,
            days=days,
            start_days_ago=start_days_ago,
            end_days_ago=end_days_ago,
            date=date,
            start_date=start_date,
            end_date=end_date,
            keywords_file=keywords_file,
            topic=topic,
            provider=provider,
            model_override=model,
            write_google_sheets=write_google_sheets,
            google_sheets_spreadsheet_id=google_sheets_spreadsheet_id,
            google_sheets_summary_worksheet=google_sheets_summary_worksheet,
            google_sheets_keywords_worksheet=google_sheets_keywords_worksheet,
            twitter_post_limit=int(os.getenv("TWITTER_POST_LIMIT", "100")),
            twitter_reply_limit=int(os.getenv("TWITTER_REPLY_LIMIT", "300")),
            twitter_max_conversations=int(os.getenv("TWITTER_MAX_CONVERSATIONS", "20")),
        )
    )


class _BearerTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request, call_next):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[len("Bearer "):] != self._token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


def create_http_app():
    asgi_app = mcp.http_app()
    api_key = os.getenv("MCP_API_KEY", "").strip()
    if api_key:
        asgi_app.add_middleware(_BearerTokenMiddleware, token=api_key)
    return asgi_app


def run() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
    if transport == "http":
        host = os.getenv("MCP_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_PORT", os.getenv("PORT", "8001")))
        uvicorn.run(create_http_app(), host=host, port=port)
        return

    mcp.run()


if __name__ == "__main__":
    run()