import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

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


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze weekly subreddit sentiment for MistralAI and products."
    )
    parser.add_argument("--subreddit", default=DEFAULT_SUBREDDIT)
    parser.add_argument("--topic", default=DEFAULT_ANALYSIS_TOPIC)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--start-days-ago", type=int, default=None)
    parser.add_argument("--end-days-ago", type=int, default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--keywords-file", default="keywords.txt")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--provider",
        choices=["mistral", "claude"],
        default=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument(
        "--write-google-sheets",
        action="store_true",
        default=env_flag("GOOGLE_SHEETS_WRITE", default=False),
    )
    parser.add_argument(
        "--google-sheets-spreadsheet-id",
        default=os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID),
    )
    parser.add_argument(
        "--google-sheets-summary-worksheet",
        default=os.getenv("GOOGLE_SHEETS_SUMMARY_WORKSHEET", DEFAULT_SUMMARY_WORKSHEET),
    )
    parser.add_argument(
        "--google-sheets-keywords-worksheet",
        default=os.getenv("GOOGLE_SHEETS_KEYWORDS_WORKSHEET", DEFAULT_KEYWORDS_WORKSHEET),
    )
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> AnalysisOptions:
    return AnalysisOptions(
        subreddit=args.subreddit,
        days=args.days,
        start_days_ago=args.start_days_ago,
        end_days_ago=args.end_days_ago,
        date=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
        keywords_file=args.keywords_file,
        topic=args.topic,
        provider=args.provider,
        model_override=args.model,
        reddit_post_limit=int(os.getenv("REDDIT_POST_LIMIT", "300")),
        reddit_comment_limit=int(os.getenv("REDDIT_COMMENT_LIMIT", "1500")),
        reddit_crawl_concurrency=int(os.getenv("REDDIT_CRAWL_CONCURRENCY", "4")),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "").strip() or DEFAULT_PUBLIC_REDDIT_USER_AGENT,
        write_google_sheets=args.write_google_sheets,
        google_sheets_spreadsheet_id=args.google_sheets_spreadsheet_id,
        google_sheets_summary_worksheet=args.google_sheets_summary_worksheet,
        google_sheets_keywords_worksheet=args.google_sheets_keywords_worksheet,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    result = run_analysis(build_options(args))
    output = json.dumps(result, indent=2)
    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()