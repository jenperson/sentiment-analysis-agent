import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

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


class AnalyzeRequest(BaseModel):
    subreddit: str = DEFAULT_SUBREDDIT
    days: int = 7
    start_days_ago: int | None = None
    end_days_ago: int | None = None
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    keywords_file: str = "keywords.txt"
    topic: str = DEFAULT_ANALYSIS_TOPIC
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, pattern="^(mistral|claude)$")
    model: str = ""
    write_google_sheets: bool = False
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID
    google_sheets_summary_worksheet: str = DEFAULT_SUMMARY_WORKSHEET
    google_sheets_keywords_worksheet: str = DEFAULT_KEYWORDS_WORKSHEET
    slack_webhook_url: str = ""


class DiscordAnalyzeRequest(BaseModel):
    guild_id: int
    channel_ids: list[int] | None = None
    days: int = 7
    start_days_ago: int | None = None
    end_days_ago: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    keywords_file: str = "keywords.txt"
    topic: str = DEFAULT_ANALYSIS_TOPIC
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, pattern="^(mistral|claude)$")
    model: str = ""
    write_google_sheets: bool = False
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID
    google_sheets_summary_worksheet: str = "discord_sentiment_summary"
    google_sheets_keywords_worksheet: str = "discord_keyword_mentions"


class TwitterAnalyzeRequest(BaseModel):
    query: str = DEFAULT_TWITTER_QUERY
    days: int = 7
    start_days_ago: int | None = None
    end_days_ago: int | None = None
    date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    keywords_file: str = "keywords.txt"
    topic: str = DEFAULT_ANALYSIS_TOPIC
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, pattern="^(mistral|claude)$")
    model: str = ""
    write_google_sheets: bool = False
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID
    google_sheets_summary_worksheet: str = "twitter_sentiment_summary"
    google_sheets_keywords_worksheet: str = "twitter_keyword_mentions"


app = FastAPI(title="Mistral Reddit Sentiment API", version="0.1.0")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(key: str | None = Security(_api_key_header)) -> None:
    required = os.getenv("API_KEY", "").strip()
    if not required:
        return  # auth disabled — API_KEY not set
    if key != required:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def request_to_options(request: AnalyzeRequest) -> AnalysisOptions:
    return AnalysisOptions(
        subreddit=request.subreddit,
        days=request.days,
        start_days_ago=request.start_days_ago,
        end_days_ago=request.end_days_ago,
        date=request.date,
        start_date=request.start_date,
        end_date=request.end_date,
        keywords_file=request.keywords_file,
        topic=request.topic,
        provider=request.provider,
        model_override=request.model,
        reddit_post_limit=int(os.getenv("REDDIT_POST_LIMIT", "300")),
        reddit_comment_limit=int(os.getenv("REDDIT_COMMENT_LIMIT", "1500")),
        reddit_crawl_concurrency=int(os.getenv("REDDIT_CRAWL_CONCURRENCY", "4")),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "").strip() or DEFAULT_PUBLIC_REDDIT_USER_AGENT,
        write_google_sheets=request.write_google_sheets,
        google_sheets_spreadsheet_id=request.google_sheets_spreadsheet_id,
        google_sheets_summary_worksheet=request.google_sheets_summary_worksheet,
        google_sheets_keywords_worksheet=request.google_sheets_keywords_worksheet,
        slack_webhook_url=request.slack_webhook_url,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", dependencies=[Depends(_verify_api_key)])
def analyze(request: AnalyzeRequest) -> dict:
    return run_analysis(request_to_options(request))


@app.post("/analyze-discord", dependencies=[Depends(_verify_api_key)])
def analyze_discord(request: DiscordAnalyzeRequest) -> dict:
    return run_discord_analysis(
        guild_id=request.guild_id,
        channel_ids=request.channel_ids,
        days=request.days,
        start_days_ago=request.start_days_ago,
        end_days_ago=request.end_days_ago,
        start_date=request.start_date,
        end_date=request.end_date,
        topic=request.topic,
        provider=request.provider,
        model_override=request.model,
        keywords_file=request.keywords_file,
        write_google_sheets=request.write_google_sheets,
        google_sheets_spreadsheet_id=request.google_sheets_spreadsheet_id,
        google_sheets_summary_worksheet=request.google_sheets_summary_worksheet,
        google_sheets_keywords_worksheet=request.google_sheets_keywords_worksheet,
    )


@app.post("/analyze-twitter", dependencies=[Depends(_verify_api_key)])
def analyze_twitter(request: TwitterAnalyzeRequest) -> dict:
    return run_twitter_analysis(
        TwitterAnalysisOptions(
            query=request.query,
            days=request.days,
            start_days_ago=request.start_days_ago,
            end_days_ago=request.end_days_ago,
            date=request.date,
            start_date=request.start_date,
            end_date=request.end_date,
            keywords_file=request.keywords_file,
            topic=request.topic,
            provider=request.provider,
            model_override=request.model,
            write_google_sheets=request.write_google_sheets,
            google_sheets_spreadsheet_id=request.google_sheets_spreadsheet_id,
            google_sheets_summary_worksheet=request.google_sheets_summary_worksheet,
            google_sheets_keywords_worksheet=request.google_sheets_keywords_worksheet,
            twitter_post_limit=int(os.getenv("TWITTER_POST_LIMIT", "100")),
            twitter_reply_limit=int(os.getenv("TWITTER_REPLY_LIMIT", "300")),
            twitter_max_conversations=int(os.getenv("TWITTER_MAX_CONVERSATIONS", "20")),
        )
    )


def run() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", os.getenv("PORT", "8000")))
    uvicorn.run("mistral_sentiment_app.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()