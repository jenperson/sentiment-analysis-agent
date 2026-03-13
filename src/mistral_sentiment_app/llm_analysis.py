import json
import os
import re
from collections.abc import Iterable

import requests
from anthropic import Anthropic

from mistral_sentiment_app.models import CommentRecord, PostRecord

DEFAULT_LLM_PROVIDER = "mistral"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_MISTRAL_MODEL = "mistral-medium-2508"
DEFAULT_ANALYSIS_TOPIC = "Mistral AI and its products"
MISTRAL_CHAT_COMPLETIONS_URL = "https://api.mistral.ai/v1/chat/completions"


def get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_analysis_config(provider: str, model_override: str) -> tuple[str, str]:
    normalized_provider = provider.strip().lower() or DEFAULT_LLM_PROVIDER
    if normalized_provider not in {"mistral", "claude"}:
        raise RuntimeError(f"Unsupported provider: {provider}")

    if model_override.strip():
        return normalized_provider, model_override.strip()

    if normalized_provider == "mistral":
        return normalized_provider, os.getenv("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL)

    return normalized_provider, os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)


def serialize_for_analysis(
    posts: list[PostRecord], comments: list[CommentRecord]
) -> dict[str, list[dict]]:
    max_items = int(os.getenv("CLAUDE_MAX_ITEMS", os.getenv("LLM_MAX_ITEMS", "250")))
    max_chars = int(
        os.getenv("CLAUDE_MAX_CHARS_PER_ITEM", os.getenv("LLM_MAX_CHARS_PER_ITEM", "1200"))
    )

    serialized_posts = []
    for post in sorted(posts, key=lambda item: item.score, reverse=True)[:max_items]:
        serialized_posts.append(
            {
                "type": "post",
                "id": post.id,
                "score": post.score,
                "link": post.permalink,
                "text": (f"Title: {post.title}\nBody: {post.body}")[:max_chars],
            }
        )

    serialized_comments = []
    for comment in sorted(comments, key=lambda item: item.score, reverse=True)[:max_items]:
        serialized_comments.append(
            {
                "type": "comment",
                "id": comment.id,
                "score": comment.score,
                "link": comment.permalink,
                "text": comment.body[:max_chars],
            }
        )

    return {"posts": serialized_posts, "comments": serialized_comments}


def build_analysis_prompt(
    posts: list[PostRecord],
    comments: list[CommentRecord],
    subreddit_name: str,
    window_label: str,
    topic: str = DEFAULT_ANALYSIS_TOPIC,
) -> tuple[str, str]:
    dataset = serialize_for_analysis(posts, comments)
    sentiment_scale = {
        "1": "Very negative, someone is beyond frustrated and unhappy with the brand or products",
        "2": "Somewhat negative, someone is not having a good experience and is expressing frustration, but not hatred",
        "3": "Neutral, someone has a mix of positive and negative things to say leading to a balance, or does not express strong feelings one way or another",
        "4": "Somewhat positive, someone expresses satisfaction with the product or uses it successfully",
        "5": "Very positive, someone is praising the product, and may want to convince others to use it",
    }

    system_prompt = "You are a precise sentiment analyst. Return strict JSON only with no markdown."
    user_prompt = {
        "task": f"Analyze sentiment around {topic} from subreddit data.",
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
    return system_prompt, json.dumps(user_prompt)


def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise RuntimeError("LLM response did not contain JSON.")
    return json.loads(match.group(0))


def extract_text_content(content: str | list[dict] | None) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "\n".join(text_parts)
    return ""


def analyze_with_claude(system_prompt: str, user_prompt: str, model: str) -> dict:
    api_key = get_required_env("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1800,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_chunks = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text_chunks.append(block.text)

    parsed = extract_json_object("\n".join(text_chunks))
    if "average_sentiment" not in parsed or "summary" not in parsed:
        raise RuntimeError("Claude response missing required fields.")
    return parsed


def analyze_with_mistral(system_prompt: str, user_prompt: str, model: str) -> dict:
    api_key = get_required_env("MISTRAL_API_KEY")
    response = requests.post(
        MISTRAL_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 1800,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("Mistral response did not include any choices.")

    content = choices[0].get("message", {}).get("content")
    parsed = extract_json_object(extract_text_content(content))
    if "average_sentiment" not in parsed or "summary" not in parsed:
        raise RuntimeError("Mistral response missing required fields.")
    return parsed


def analyze_sentiment(
    posts: list[PostRecord],
    comments: list[CommentRecord],
    subreddit_name: str,
    window_label: str,
    provider: str,
    model_override: str,
    topic: str = DEFAULT_ANALYSIS_TOPIC,
) -> tuple[dict, str, str]:
    resolved_provider, resolved_model = resolve_analysis_config(provider, model_override)
    system_prompt, user_prompt = build_analysis_prompt(
        posts=posts,
        comments=comments,
        subreddit_name=subreddit_name,
        window_label=window_label,
        topic=topic,
    )

    if resolved_provider == "mistral":
        sentiment = analyze_with_mistral(system_prompt, user_prompt, resolved_model)
    else:
        sentiment = analyze_with_claude(system_prompt, user_prompt, resolved_model)

    return sentiment, resolved_provider, resolved_model