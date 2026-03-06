"""Tests for JSON output adapter."""

from __future__ import annotations

import json

from alchemy.outputs.json_output import JSONOutputAdapter


def test_json_output_writes_jsonl(tmp_path, sample_plan):
    path = str(tmp_path / "test_output")
    adapter = JSONOutputAdapter(path)
    samples = [
        {"question": "Q1?", "answer": "A1", "difficulty": "easy"},
        {"question": "Q2?", "answer": "A2", "difficulty": "hard"},
    ]
    result_path = adapter.write(samples, sample_plan)
    assert result_path.endswith(".jsonl")

    with open(result_path) as f:
        lines = f.readlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "Q1?"
