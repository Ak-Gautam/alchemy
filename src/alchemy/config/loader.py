"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from .settings import GlobalConfig, PipelineConfig


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline configuration from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**(data or {}))


def load_global_config(path: str | Path | None = None) -> GlobalConfig:
    """Load global config from ~/.config/alchemy/config.yaml or a given path."""
    if path is None:
        path = Path.home() / ".config" / "alchemy" / "config.yaml"
    path = Path(path)
    if not path.exists():
        return GlobalConfig()
    with open(path) as f:
        data = yaml.safe_load(f)
    return GlobalConfig(**(data or {}))
