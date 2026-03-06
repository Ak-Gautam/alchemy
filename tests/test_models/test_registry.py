"""Tests for model provider registry."""

from __future__ import annotations

import pytest

from alchemy.models.registry import create_provider


def test_create_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider type"):
        create_provider("nonexistent", model_id="foo")


def test_create_provider_mlx_returns_instance():
    # Don't actually load a model, just verify the class is instantiated
    provider = create_provider("mlx", model_id="test-model")
    assert provider.model_name() == "test-model"


def test_create_provider_openai_returns_instance():
    provider = create_provider("openai", model_id="gpt-4o")
    assert provider.model_name() == "gpt-4o"


def test_create_provider_google_returns_instance():
    provider = create_provider("google", model_id="gemini-pro")
    assert provider.model_name() == "gemini-pro"
