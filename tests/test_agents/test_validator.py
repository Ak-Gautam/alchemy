"""Tests for validator agent."""

from __future__ import annotations

import json

from alchemy.agents.validator import ValidatorAgent


def test_validator_parses_results(mock_provider, sample_plan):
    validation = json.dumps([
        {"index": 0, "is_valid": True, "score": 0.95, "issues": []},
        {"index": 1, "is_valid": False, "score": 0.3, "issues": ["too short"]},
    ])
    provider = mock_provider([validation])
    agent = ValidatorAgent(provider)

    samples = [
        {"question": "What is H2O?", "answer": "Water", "difficulty": "easy"},
        {"question": "Q?", "answer": "A", "difficulty": "x"},
    ]
    results = agent.validate_batch(sample_plan, samples)
    assert len(results) == 2
    assert results[0].is_valid is True
    assert results[0].score == 0.95
    assert results[1].is_valid is False
    assert "too short" in results[1].issues
