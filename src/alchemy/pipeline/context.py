"""Pipeline execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alchemy.pipeline.plan import GenerationPlan


@dataclass
class PipelineContext:
    """Mutable shared state carried through all pipeline steps."""

    user_prompt: str
    plan: GenerationPlan | None = None
    raw_samples: list[dict[str, Any]] = field(default_factory=list)
    validated_samples: list[dict[str, Any]] = field(default_factory=list)
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)
    output_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
