"""Tests for pipeline engine."""

from __future__ import annotations

import json

from alchemy.config.settings import PipelineConfig
from alchemy.outputs.base import OutputAdapter
from alchemy.pipeline.engine import PipelineEngine


def test_pipeline_engine_full_run(mock_provider, sample_plan, tmp_path):
    """Test the full pipeline with mocked providers."""
    plan_json = sample_plan.model_dump_json()

    sample_batch = json.dumps([
        {"question": "What is H2O?", "answer": "Water is H2O.", "difficulty": "easy"},
        {"question": "Explain photosynthesis in detail", "answer": "Plants convert sunlight to energy.", "difficulty": "medium"},
    ])

    validation_result = json.dumps([
        {"index": 0, "is_valid": True, "score": 0.9, "issues": []},
        {"index": 1, "is_valid": True, "score": 0.85, "issues": []},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider([validation_result])

    output_path = str(tmp_path / "test_output.jsonl")
    output_adapter = OutputAdapter.create("json", output_path)

    # Override the plan's num_samples to match our single batch
    config = PipelineConfig(num_samples=2, batch_size=5)
    engine = PipelineEngine(planner, generator, validator, output_adapter, config)
    ctx = engine.run("Generate chemistry Q&A", num_samples=2)

    assert ctx.plan is not None
    assert ctx.plan.dataset_name == "test_qa"
    assert len(ctx.validated_samples) == 2
    assert ctx.output_path is not None
