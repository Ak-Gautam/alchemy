"""Tests for robust JSON parsing utilities."""

from __future__ import annotations

import pytest

from alchemy.utils.json_parsing import extract_json_payload, parse_json_payload


def test_parse_json_payload_from_fenced_block():
    text = "```json\n{\"a\": 1, \"b\": [1,2,3]}\n```"
    data = parse_json_payload(text)
    assert data["a"] == 1
    assert data["b"] == [1, 2, 3]


def test_parse_json_payload_after_think_block():
    text = "<think>internal reasoning</think>\n{\"ok\": true}"
    data = parse_json_payload(text)
    assert data["ok"] is True


def test_extract_first_json_span_from_noisy_text():
    text = "Here is data: {\"x\": {\"y\": \"value with } brace\"}} trailing text"
    extracted = extract_json_payload(text)
    assert extracted == "{\"x\": {\"y\": \"value with } brace\"}}"


def test_parse_json_payload_raises_when_missing():
    with pytest.raises(ValueError, match="No JSON object/array payload"):
        parse_json_payload("no json here")
