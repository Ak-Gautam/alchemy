"""Tests for configuration models."""

from __future__ import annotations

from alchemy.config.settings import GlobalConfig, ModelConfig, PipelineConfig


def test_model_config():
    mc = ModelConfig(provider_type="openai", model_id="gpt-4o")
    assert mc.provider_type == "openai"
    assert mc.options == {}


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.output_format == "huggingface"
    assert config.num_samples == 100
    assert config.planner_model.provider_type == "mlx"


def test_global_config_defaults():
    config = GlobalConfig()
    assert config.log_level == "INFO"
    assert config.default_output_dir == "./output"
