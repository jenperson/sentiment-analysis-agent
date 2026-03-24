"""
Discord sentiment analysis service.

Fetches messages from a Discord server over a given time period,
analyzes sentiment using an LLM, and optionally writes results to Google Sheets.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import discord

from mistral_sentiment_app.google_sheets_export import (
    DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    DEFAULT_KEYWORDS_WORKSHEET,
    DEFAULT_SUMMARY_WORKSHEET,
    write_results_to_google_sheets,
)
from mistral_sentiment_app.llm_analysis import (
    DEFAULT_ANALYSIS_TOPIC,
    DEFAULT_LLM_PROVIDER,
    analyze_sentiment,
)
from mistral_sentiment_app.models import CommentRecord, PostRecord
from mistral_sentiment_app.service import (
    AnalysisOptions,
    build_result,
    compute_window,
    keyword_mentions,
    load_keywords,
)

DEFAULT_DISCORD_GUILD_ID: Optional[int] = None  # Will use env or request param
DEFAULT_DISCORD_CHANNELS: list[int] = []  # Empty = all channels


class DiscordClient:
    """Discord client for fetching messages."""

    def __init__(self, token: str) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self.token = token

    async def fetch_guild_messages(
        self,
        guild_id: int,
        channel_ids: Optional[list[int]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> tuple[list[PostRecord], list[CommentRecord]]:
        """
        Fetch messages from a Discord guild within a time range.

        Args:
            guild_id: Discord server ID
            channel_ids: List of channel IDs to fetch from (None = all accessible)
            start_time: Earliest message time (UTC)
            end_time: Latest message time (UTC)
            limit: Max messages to fetch per channel

        Returns:
            (messages as PostRecords, thread_replies as CommentRecords)
        """
        await self.client.login(self.token)
        try:
            guild = self.client.get_guild(guild_id)
            if not guild:
                raise ValueError(f"Guild {guild_id} not found or not accessible")

            posts: list[PostRecord] = []
            comments: list[CommentRecord] = []

            channels_to_fetch = []
            if channel_ids:
                channels_to_fetch = [guild.get_channel(cid) for cid in channel_ids]
                channels_to_fetch = [c for c in channels_to_fetch if c]
            else:
                channels_to_fetch = [
                    c for c in guild.text_channels if isinstance(c, discord.TextChannel)
                ]

            for channel in channels_to_fetch:
                try:
                    async for message in channel.history(limit=limit, oldest_first=False):
                        if start_time and message.created_at < start_time:
                            continue
                        if end_time and message.created_at > end_time:
                            continue

                        if message.author.bot:
                            continue

                        content = message.content
                        if not content and message.embeds:
                            content = f"(embed) {message.embeds[0].title or ''}"
                        if not content:
                            continue

                        post = PostRecord(
                            id=str(message.id),
                            title=f"#{channel.name} by {message.author.name}",
                            body=content,
                            score=len(message.reactions),
                            created_utc=message.created_at.timestamp(),
                            permalink=message.jump_url,
                        )
                        posts.append(post)

                        if message.thread:
                            async for reply in message.thread.history(limit=100):
                                if reply.author.bot:
                                    continue
                                reply_content = reply.content
                                if not reply_content and reply.embeds:
                                    reply_content = f"(embed) {reply.embeds[0].title or ''}"
                                if not reply_content:
                                    continue

                                comment = CommentRecord(
                                    id=str(reply.id),
                                    body=reply_content,
                                    score=len(reply.reactions),
                                    created_utc=reply.created_at.timestamp(),
                                    permalink=reply.jump_url,
                                )
                                comments.append(comment)

                except discord.Forbidden:
                    continue

        finally:
            await self.client.close()

        return posts, comments


async def fetch_discord_data(
    guild_id: int,
    channel_ids: Optional[list[int]] = None,
    days: Optional[int] = None,
    start_days_ago: Optional[int] = None,
    end_days_ago: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
) -> tuple[list[PostRecord], list[CommentRecord]]:
    """
    Fetch Discord messages over a time period.

    Args:
        guild_id: Discord server ID
        channel_ids: Specific channels to fetch (None = all)
        days: Trailing days from now (default 7)
        start_days_ago: Older boundary (days ago)
        end_days_ago: Newer boundary (days ago, default 0)
        start_date: Specific start date (YYYY-MM-DD)
        end_date: Specific end date (YYYY-MM-DD)
        limit: Max messages per channel

    Returns:
        (messages, replies)
    """
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN not set")

    if start_time is None or end_time is None:
        now_utc = datetime.now(timezone.utc)

        if start_date:
            start_time = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        elif start_days_ago is not None:
            start_time = now_utc - timedelta(days=start_days_ago)
        elif days is not None:
            start_time = now_utc - timedelta(days=days)
        else:
            start_time = now_utc - timedelta(days=7)

        if end_date:
            end_time = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        elif end_days_ago is not None:
            end_time = now_utc - timedelta(days=end_days_ago)
        else:
            end_time = now_utc

    client = DiscordClient(token)
    return await client.fetch_guild_messages(
        guild_id=guild_id,
        channel_ids=channel_ids,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )


def run_discord_analysis(
    guild_id: int,
    channel_ids: Optional[list[int]] = None,
    days: int = 7,
    start_days_ago: Optional[int] = None,
    end_days_ago: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    topic: str = DEFAULT_ANALYSIS_TOPIC,
    provider: str = DEFAULT_LLM_PROVIDER,
    model_override: str = "",
    keywords_file: str = "keywords.txt",
    write_google_sheets: bool = False,
    google_sheets_spreadsheet_id: str = DEFAULT_GOOGLE_SHEETS_SPREADSHEET_ID,
    google_sheets_summary_worksheet: str = DEFAULT_SUMMARY_WORKSHEET,
    google_sheets_keywords_worksheet: str = DEFAULT_KEYWORDS_WORKSHEET,
) -> dict:
    """
    Analyze Discord sentiment for a guild over a time period.

    Returns a JSON result with sentiment analysis, summary, top messages, and keywords.
    """
    import asyncio

    keywords = load_keywords(Path(keywords_file))
    window_start, window_end, window_label = compute_window(
        AnalysisOptions(
            days=days,
            start_days_ago=start_days_ago,
            end_days_ago=end_days_ago,
            date=None,
            start_date=start_date,
            end_date=end_date,
        )
    )

    messages, replies = asyncio.run(
        fetch_discord_data(
            guild_id=guild_id,
            channel_ids=channel_ids,
            start_time=window_start,
            end_time=window_end,
        )
    )

    mentions = keyword_mentions(keywords=keywords, posts=messages, comments=replies)

    if not messages and not replies:
        sentiment = {
            "average_sentiment": None,
            "summary": "No Discord messages or thread replies were found in the selected window.",
            "method_notes": "LLM analysis skipped because the Discord fetch returned an empty dataset.",
        }
        result = build_result(
            subreddit_name=f"Discord Guild {guild_id}",
            window_start=window_start,
            window_end=window_end,
            window_label=window_label,
            posts=messages,
            comments=replies,
            sentiment=sentiment,
            mentions=mentions,
            analysis_provider=provider,
            analysis_model=model_override or "",
            top_posts_method="upvotes",
        )
        result["guild_id"] = guild_id

        if write_google_sheets:
            try:
                gs_result = write_results_to_google_sheets(
                    result=result,
                    spreadsheet_id=google_sheets_spreadsheet_id,
                    summary_worksheet_name=google_sheets_summary_worksheet,
                    keywords_worksheet_name=google_sheets_keywords_worksheet,
                )
                result["google_sheets_export"] = gs_result
            except Exception as exc:  # noqa: BLE001
                result["google_sheets_export"] = {"status": "failed", "error": str(exc)}

        return result

    sentiment, analysis_provider, analysis_model = analyze_sentiment(
        posts=messages,
        comments=replies,
        subreddit_name=f"Discord Guild {guild_id}",
        window_label=window_label,
        provider=provider,
        model_override=model_override,
        topic=topic,
    )
    result = build_result(
        subreddit_name=f"Discord Guild {guild_id}",
        window_start=window_start,
        window_end=window_end,
        window_label=window_label,
        posts=messages,
        comments=replies,
        sentiment=sentiment,
        mentions=mentions,
        analysis_provider=analysis_provider,
        analysis_model=analysis_model,
        top_posts_method="upvotes",
    )
    result["guild_id"] = guild_id

    if write_google_sheets:
        try:
            gs_result = write_results_to_google_sheets(
                result=result,
                spreadsheet_id=google_sheets_spreadsheet_id,
                summary_worksheet_name=google_sheets_summary_worksheet,
                keywords_worksheet_name=google_sheets_keywords_worksheet,
            )
            result["google_sheets_export"] = gs_result
        except Exception as exc:  # noqa: BLE001
            result["google_sheets_export"] = {"status": "failed", "error": str(exc)}

    return result
