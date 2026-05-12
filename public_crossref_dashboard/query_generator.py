"""OpenAI-powered query generation utilities."""

from __future__ import annotations

import re
from typing import List

from openai import OpenAI


def _extract_candidate_lines(text: str) -> List[str]:
    """Parse model text output into candidate query lines."""
    lines = []
    for raw in text.splitlines():
        cleaned = raw.strip()
        cleaned = re.sub(r"^\d+[\).\-\s]+", "", cleaned)
        cleaned = re.sub(r"^[\-\*\u2022]\s*", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip("'")
        if cleaned:
            lines.append(cleaned)
    return lines


def _normalize_queries(candidates: List[str], target_count: int) -> List[str]:
    """Normalize and deduplicate query candidates."""
    seen = set()
    normalized = []
    for item in candidates:
        text = re.sub(r"\s+", " ", item).strip(" ,;.")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= target_count:
            break
    return normalized


def generate_queries_openai(
    *,
    api_key: str,
    model: str,
    research_query: str,
    system_prompt: str,
    target_count: int = 20,
) -> List[str]:
    """
    Generate query terms/phrases using OpenAI.

    Returns up to `target_count` normalized unique queries.
    Retries once with a stricter instruction if fewer than target returned.
    """
    client = OpenAI(api_key=api_key)

    user_prompt = (
        f"Research topic: {research_query}\n\n"
        f"Return exactly {target_count} unique search queries as a plain newline list. "
        "Include a mix of single terms and 2-3 word phrases. "
        "Do not add explanations."
    )

    def _call_model(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content or ""

    first_output = _call_model(user_prompt)
    queries = _normalize_queries(_extract_candidate_lines(first_output), target_count)

    if len(queries) < target_count:
        retry_prompt = (
            user_prompt
            + "\n\nYour previous answer had too few items. Return EXACTLY "
            + str(target_count)
            + " unique items, one per line."
        )
        retry_output = _call_model(retry_prompt)
        queries = _normalize_queries(
            _extract_candidate_lines(first_output) + _extract_candidate_lines(retry_output),
            target_count,
        )

    return queries


def fetch_available_models_openai(*, api_key: str) -> List[str]:
    """Fetch all available model IDs for the provided OpenAI API key."""
    client = OpenAI(api_key=api_key)
    response = client.models.list()
    model_ids = sorted({model.id for model in response.data if getattr(model, "id", None)})
    return model_ids
