"""Configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a single model provider."""

    provider_type: str
    model_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Full configuration for a pipeline run."""

    planner_model: ModelConfig = ModelConfig(
        provider_type="mlx",
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
    )
    generator_model: ModelConfig = ModelConfig(
        provider_type="mlx",
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
    )
    validator_model: ModelConfig = ModelConfig(
        provider_type="mlx",
        model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
    )
    output_format: str = "huggingface"
    output_path: str = "./output"
    num_samples: int = 100
    batch_size: int = 10
    min_quality_score: float = 0.7


class GlobalConfig(BaseModel):
    """Top-level application configuration."""

    api_keys_file: str = "~/.config/alchemy/keys.yaml"
    default_output_dir: str = "./output"
    log_level: str = "INFO"
    default_pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
