"""
Slack sentiment analysis service.

Formats sentiment analysis results and sends them to Slack.
"""

import os
from typing import Optional

import requests


def format_sentiment_message(result: dict, topic: str = "Sentiment Analysis") -> dict:
    """
    Format sentiment analysis results into a Slack message.

    Args:
        result: The sentiment analysis result dict from run_analysis()
        topic: Topic name for the message title

    Returns:
        A dict containing Slack message payload
    """
    subreddit = result.get("subreddit", "Unknown")
    window = result.get("window", {})
    window_label = window.get("label", "Unknown")
    counts = result.get("counts", {})
    sentiment = result.get("average_sentiment")
    summary = result.get("summary_of_week", "No summary available")
    top_posts = result.get("top_3_posts_by_upvotes", [])
    mentions = result.get("keyword_mentions", {})
    analysis = result.get("analysis", {})

    # Determine sentiment color and emoji
    sentiment_color = "#808080"  # Gray default
    sentiment_emoji = "➖"
    if sentiment is not None:
        if sentiment > 0.3:
            sentiment_color = "#36a64f"  # Green
            sentiment_emoji = "😊"
        elif sentiment < -0.3:
            sentiment_color = "#e74c3c"  # Red
            sentiment_emoji = "😞"
        else:
            sentiment_color = "#f39c12"  # Orange
            sentiment_emoji = "😐"

    # Build blocks
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📊 {topic} Report for r/{subreddit}"},
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Period:*\n{window_label}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Posts:*\n{counts.get('posts', 0)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Comments:*\n{counts.get('comments', 0)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Sentiment:*\n{sentiment_emoji}",
                },
            ],
        },
        {
            "type": "section",
            "block_id": "sentiment_block",
            "text": {
                "type": "mrkdwn",
                "text": f"*Overall Sentiment Score:* {sentiment if sentiment is not None else 'N/A'}",
            },
            "accessory": {
                "type": "context_element",
                "text": {"type": "mrkdwn", "text": sentiment_emoji},
            },
        },
    ]

    # Add sentiment summary
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Summary:*\n{summary}"},
        }
    )

    # Add keyword mentions if available
    if mentions:
        mentions_text = "*Keyword Mentions:*\n"
        for keyword, data in mentions.items():
            posts = data.get("post_count", 0)
            comments = data.get("comment_count", 0)
            mentions_text += (
                f"• *{keyword}*: {posts} posts, {comments} comments\n"
            )
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": mentions_text}})

    # Add top posts
    if top_posts:
        top_posts_text = "*Top 3 Posts by Upvotes:*\n"
        for i, post in enumerate(top_posts[:3], 1):
            title = post.get("title", "No title")
            score = post.get("score", 0)
            url = post.get("url", "#")
            top_posts_text += f"{i}. <{url}|{title}> ({score} upvotes)\n"
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": top_posts_text}})

    # Add analysis metadata
    provider = analysis.get("provider", "Unknown")
    model = analysis.get("model", "Unknown")
    metadata_text = f"*Analysis:* {provider} | *Model:* {model}"
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": metadata_text}})

    # Create attachments with color coding
    attachments = [
        {
            "color": sentiment_color,
            "blocks": blocks,
        }
    ]

    return {
        "text": f"Sentiment Analysis Report for r/{subreddit}",
        "attachments": attachments,
    }


def send_slack_message(
    webhook_url: str,
    message_payload: dict,
    timeout: int = 10,
) -> bool:
    """
    Send a formatted message to Slack via webhook.

    Args:
        webhook_url: Slack webhook URL
        message_payload: The message payload dict (from format_sentiment_message)
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    if not webhook_url:
        print("Warning: Slack webhook URL not provided")
        return False

    try:
        response = requests.post(webhook_url, json=message_payload, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack message: {e}")
        return False


def send_analysis_to_slack(
    result: dict,
    webhook_url: Optional[str] = None,
    topic: str = "Sentiment Analysis",
) -> dict:
    """
    Format and send sentiment analysis results to Slack.

    Args:
        result: The sentiment analysis result dict
        webhook_url: Slack webhook URL (uses env var if not provided)
        topic: Topic name for the message title

    Returns:
        A dict with status information about the Slack export
    """
    webhook = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "").strip()

    if not webhook:
        return {
            "success": False,
            "message": "Slack webhook URL not configured",
        }

    message = format_sentiment_message(result, topic)
    success = send_slack_message(webhook, message)

    return {
        "success": success,
        "message": "Message sent to Slack" if success else "Failed to send Slack message",
        "webhook_url_hash": hash(webhook) % 10000 if webhook else None,  # Sanitized for logging
    }
