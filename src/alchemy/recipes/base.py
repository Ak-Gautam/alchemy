"""Recipe interfaces for dataset-specific planning defaults."""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field

from alchemy.quality.language_constraints import validate_language_constraints
from alchemy.spec.plan import GenerationPlan, PlanDefaults, VariationSpec


class LanguageConstraints(BaseModel):
    """Constraints for code fields in recipe-defined datasets."""

    language: str
    code_field: str = "code"
    disallowed_substrings: list[str] = Field(default_factory=list)
    required_substrings: list[str] = Field(default_factory=list)


class BaseRecipe(abc.ABC):
    """Base class for reusable dataset recipe definitions."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def row_schema(self) -> dict[str, Any]:
        ...

    @property
    def quality_rubric(self) -> list[str]:
        return []

    @property
    def variation_spec(self) -> VariationSpec:
        return VariationSpec()

    @property
    def metadata(self) -> dict[str, Any]:
        return {"recipe": self.name}

    @property
    def language_constraints(self) -> LanguageConstraints | None:
        return None

    def build_plan(self, *, num_rows: int = 1000, batch_size: int = 10) -> GenerationPlan:
        return GenerationPlan(
            dataset_name=self.name,
            description=self.description,
            row_schema=self.row_schema,
            defaults=PlanDefaults(num_rows=num_rows, batch_size=batch_size),
            variation_spec=self.variation_spec,
            quality_rubric=self.quality_rubric,
            metadata=self.metadata,
        )

    def validate_row_rules(self, row: dict[str, Any]) -> list[str]:
        constraints = self.language_constraints
        if constraints is None:
            return []
        return validate_language_constraints(
            row,
            code_field=constraints.code_field,
            language=constraints.language,
            disallowed_substrings=constraints.disallowed_substrings,
            required_substrings=constraints.required_substrings,
        )
