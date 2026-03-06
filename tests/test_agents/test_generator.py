"""Tests for generator agent."""

from __future__ import annotations

import json

from alchemy.agents.generator import GeneratorAgent


def test_generator_parses_array(mock_provider, sample_plan):
    samples = json.dumps([
        {"question": "What is H2O?", "answer": "Water", "difficulty": "easy"},
        {"question": "Explain DNA replication", "answer": "DNA copies itself", "difficulty": "hard"},
    ])
    provider = mock_provider([samples])
    agent = GeneratorAgent(provider)
    result = agent.generate_batch(sample_plan, batch_size=2, batch_index=0)
    assert len(result) == 2
    assert result[0]["question"] == "What is H2O?"


def test_generator_parses_wrapped_object(mock_provider, sample_plan):
    samples = json.dumps({
        "samples": [
            {"question": "Q1?", "answer": "A1", "difficulty": "easy"},
        ]
    })
    provider = mock_provider([samples])
    agent = GeneratorAgent(provider)
    result = agent.generate_batch(sample_plan, batch_size=1, batch_index=0)
    assert len(result) == 1
