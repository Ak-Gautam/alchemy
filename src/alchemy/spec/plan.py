"""Schema-first plan models used by the next-generation pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PlanDefaults(BaseModel):
    """Default run sizing inferred by the planner."""

    num_rows: int = Field(default=1000, ge=1)
    batch_size: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _validate_batch_size(self) -> PlanDefaults:
        if self.batch_size > self.num_rows:
            raise ValueError("batch_size cannot be larger than num_rows")
        return self


class VariationAxis(BaseModel):
    """One axis of diversity for coverage-aware generation."""

    name: str
    values: list[str] = Field(default_factory=list)
    distribution: dict[str, float] | None = None

    @model_validator(mode="after")
    def _validate_distribution(self) -> VariationAxis:
        if self.distribution is None:
            return self

        unknown_values = sorted(set(self.distribution) - set(self.values))
        if unknown_values:
            raise ValueError(
                f"distribution contains values not present in axis '{self.name}': {unknown_values}"
            )

        total = sum(self.distribution.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(
                f"distribution for axis '{self.name}' must sum to 1.0 (+/-0.001), got {total}"
            )
        return self


class VariationSpec(BaseModel):
    """Global diversity controls for planning and batch scheduling."""

    axes: list[VariationAxis] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)


class SafetySpec(BaseModel):
    """Safety policy controls applied during validation."""

    allow_pii: bool = False
    disallowed_categories: list[str] = Field(default_factory=list)


class BatchTarget(BaseModel):
    """Desired count for one value inside an axis for a generation batch."""

    axis: str
    value: str
    count: int = Field(ge=1)


class BatchSpec(BaseModel):
    """Planner/scheduler output for a single batch request."""

    batch_size: int = Field(ge=1)
    targets: list[BatchTarget] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    """Schema-first plan for dataset generation."""

    dataset_name: str
    description: str
    row_schema: dict[str, Any]
    defaults: PlanDefaults = Field(default_factory=PlanDefaults)
    variation_spec: VariationSpec = Field(default_factory=VariationSpec)
    quality_rubric: list[str] = Field(default_factory=list)
    example_rows: list[dict[str, Any]] = Field(default_factory=list)
    safety: SafetySpec = Field(default_factory=SafetySpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row_schema(self) -> GenerationPlan:
        # Keep the first increment lightweight: only enforce object schema shape.
        if self.row_schema.get("type") != "object":
            raise ValueError("row_schema.type must be 'object'")
        if "properties" not in self.row_schema:
            raise ValueError("row_schema must include 'properties'")
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationPlan:
        """Construct a plan from parsed JSON data."""
        return cls(**data)

    def schema_summary(self) -> str:
        """Short textual summary for prompt templates and logs."""
        properties = self.row_schema.get("properties", {})
        required = self.row_schema.get("required", [])
        field_names = ", ".join(sorted(properties.keys()))
        return (
            f"Dataset: {self.dataset_name}\n"
            f"Description: {self.description}\n"
            f"Fields: {field_names}\n"
            f"Required: {', '.join(required) if required else '(none)'}\n"
            f"Target rows: {self.defaults.num_rows} (batch {self.defaults.batch_size})"
        )
