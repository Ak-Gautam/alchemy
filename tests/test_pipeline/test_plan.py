"""Tests for generation plan."""

from __future__ import annotations

from alchemy.pipeline.plan import FieldSchema, GenerationPlan


def test_generation_plan_from_dict():
    data = {
        "dataset_name": "test",
        "description": "A test dataset",
        "fields": [
            {
                "name": "text",
                "field_type": "string",
                "description": "Some text",
            }
        ],
        "generation_strategy": "random",
        "num_samples": 10,
    }
    plan = GenerationPlan.from_dict(data)
    assert plan.dataset_name == "test"
    assert len(plan.fields) == 1
    assert plan.fields[0].name == "text"
    assert plan.num_samples == 10
    assert plan.batch_size == 10  # default


def test_schema_summary(sample_plan):
    summary = sample_plan.schema_summary()
    assert "test_qa" in summary
    assert "question" in summary
    assert "answer" in summary
    assert "difficulty" in summary
    assert "Diversity dimensions" in summary


def test_field_schema_constraints():
    field = FieldSchema(
        name="code",
        field_type="code",
        description="Python code",
        constraints={"language": "python", "min_length": 20},
    )
    assert field.constraints["language"] == "python"
