"""Generation plan data structures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FieldSchema(BaseModel):
    """Definition of a single field in the dynamic dataset schema."""

    name: str
    field_type: str
    description: str
    constraints: dict[str, Any] = Field(default_factory=dict)


class GenerationPlan(BaseModel):
    """Output of the Planner agent — defines everything needed to generate a dataset."""

    dataset_name: str
    description: str
    fields: list[FieldSchema]
    generation_strategy: str
    num_samples: int
    batch_size: int = 10
    example_samples: list[dict[str, Any]] = Field(default_factory=list)
    diversity_dimensions: list[str] = Field(default_factory=list)
    quality_criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationPlan:
        return cls(**data)

    def schema_summary(self) -> str:
        """Human-readable summary for inclusion in prompts."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Description: {self.description}",
            "Fields:",
        ]
        for f in self.fields:
            constraint_str = f" {f.constraints}" if f.constraints else ""
            lines.append(f"  - {f.name} ({f.field_type}): {f.description}{constraint_str}")
        if self.diversity_dimensions:
            lines.append(f"Diversity dimensions: {', '.join(self.diversity_dimensions)}")
        if self.quality_criteria:
            lines.append(f"Quality criteria: {', '.join(self.quality_criteria)}")
        return "\n".join(lines)
