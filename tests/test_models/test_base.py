"""Tests for base model types."""

from __future__ import annotations

from alchemy.models.base import GenerationConfig, GenerationResult, Message


def test_message_creation():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_generation_config_defaults():
    config = GenerationConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.stop_sequences == []


def test_generation_result():
    result = GenerationResult(text="output", model="test", usage={"prompt_tokens": 10})
    assert result.text == "output"
    assert result.usage["prompt_tokens"] == 10
