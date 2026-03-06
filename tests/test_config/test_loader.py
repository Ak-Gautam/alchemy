"""Tests for config loading."""

from __future__ import annotations

from alchemy.config.loader import load_global_config, load_pipeline_config


def test_load_pipeline_config_from_yaml(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""\
planner_model:
  provider_type: openai
  model_id: gpt-4o
  init:
    api_key: test-key
  generation:
    temperature: 0.2
    max_tokens: 1024
num_samples: 50
output_format: json
""")
    config = load_pipeline_config(config_file)
    assert config.planner_model.provider_type == "openai"
    assert config.planner_model.model_id == "gpt-4o"
    assert config.planner_model.init["api_key"] == "test-key"
    assert config.planner_model.generation.temperature == 0.2
    assert config.planner_model.generation.max_tokens == 1024
    assert config.num_samples == 50
    assert config.output_format == "json"


def test_load_pipeline_config_legacy_options_supported(tmp_path):
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text("""\
planner_model:
  provider_type: openai
  model_id: gpt-4o
  options:
    api_key: legacy-key
""")
    config = load_pipeline_config(config_file)
    assert config.planner_model.init["api_key"] == "legacy-key"


def test_load_global_config_missing_file():
    config = load_global_config("/nonexistent/path/config.yaml")
    assert config.log_level == "INFO"
