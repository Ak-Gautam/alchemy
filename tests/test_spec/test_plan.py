"""Tests for alchemy.spec.plan models."""

from __future__ import annotations

import pytest

from alchemy.spec.plan import GenerationPlan, PlanDefaults, VariationAxis


def _sample_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "completion": {"type": "string"},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        },
        "required": ["prompt", "completion"],
        "additionalProperties": False,
    }


def test_plan_defaults_valid():
    defaults = PlanDefaults()
    assert defaults.num_rows == 1000
    assert defaults.batch_size == 10


def test_plan_defaults_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        PlanDefaults(num_rows=10, batch_size=11)


def test_variation_axis_distribution_validation():
    with pytest.raises(ValueError, match="not present"):
        VariationAxis(
            name="difficulty",
            values=["easy", "medium", "hard"],
            distribution={"easy": 0.5, "unknown": 0.5},
        )

    with pytest.raises(ValueError, match="must sum to 1.0"):
        VariationAxis(
            name="difficulty",
            values=["easy", "medium", "hard"],
            distribution={"easy": 0.5, "medium": 0.2, "hard": 0.2},
        )


def test_generation_plan_from_dict_and_summary():
    plan = GenerationPlan.from_dict(
        {
            "dataset_name": "instruction_pairs",
            "description": "Instruction/completion samples",
            "row_schema": _sample_schema(),
            "quality_rubric": ["factual", "clear", "non-repetitive"],
        }
    )
    assert plan.dataset_name == "instruction_pairs"
    assert plan.defaults.num_rows == 1000
    assert "prompt" in plan.schema_summary()
    assert "Target rows: 1000" in plan.schema_summary()


def test_generation_plan_requires_object_schema():
    with pytest.raises(ValueError, match="row_schema.type"):
        GenerationPlan(
            dataset_name="bad",
            description="bad",
            row_schema={"type": "array", "items": {"type": "string"}},
        )

    with pytest.raises(ValueError, match="properties"):
        GenerationPlan(
            dataset_name="bad2",
            description="bad2",
            row_schema={"type": "object"},
        )
