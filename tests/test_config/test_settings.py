"""Tests for configuration models."""

from __future__ import annotations

from alchemy.config.settings import GlobalConfig, ModelConfig, PipelineConfig


def test_model_config():
    mc = ModelConfig(provider_type="openai", model_id="gpt-4o")
    assert mc.provider_type == "openai"
    assert mc.init == {}
    assert mc.options == {}
    assert mc.generation.max_tokens == 4096


def test_model_config_legacy_options_migrate_to_init():
    mc = ModelConfig(provider_type="openai", model_id="gpt-4o", options={"api_key": "x"})
    assert mc.init == {"api_key": "x"}
    assert mc.options == {"api_key": "x"}


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.output_format == "huggingface"
    assert config.num_samples == 100
    assert config.planner_model.provider_type == "mlx"
    assert config.planner_model.generation.temperature == 0.7
    assert config.postprocess_hf_from_jsonl is False
    assert config.postprocess_hf_output_path is None


def test_global_config_defaults():
    config = GlobalConfig()
    assert config.log_level == "INFO"
    assert config.default_output_dir == "./output"
