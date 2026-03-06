"""Robust JSON extraction/parsing utilities for LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def build_json_repair_message(error: Exception) -> str:
    """Return a structured retry prompt when JSON parsing fails."""
    return (
        "Your previous response could not be parsed as valid JSON. "
        f"Parsing error: {error}. "
        "Return ONLY valid JSON (object or array), without markdown fences or explanation."
    )


def parse_json_payload(text: str) -> Any:
    """Extract and parse the first valid JSON object/array from model output text."""
    payload = extract_json_payload(text)
    return json.loads(payload)


def extract_json_payload(text: str) -> str:
    """Extract the first JSON object or array span from noisy model output."""
    cleaned = _normalize_text(text)
    extracted = _extract_first_json_span(cleaned)
    if extracted is None:
        raise ValueError("No JSON object/array payload found in response")
    return extracted


def _normalize_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = _THINK_RE.sub("", cleaned).strip()

    fenced = _FENCE_RE.search(cleaned)
    if fenced is not None:
        cleaned = fenced.group(1).strip()
    return cleaned


def _extract_first_json_span(text: str) -> str | None:
    start = -1
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start == -1:
        return None

    stack: list[str] = []
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char in "}]":
            if not stack:
                return None
            opening = stack.pop()
            if (opening, char) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                return text[start : index + 1].strip()
    return None
