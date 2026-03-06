"""Tests for pipeline engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alchemy.config.settings import PipelineConfig
from alchemy.exceptions import GenerationError
from alchemy.models.base import GenerationConfig
from alchemy.outputs.base import OutputAdapter
from alchemy.pipeline.engine import PipelineEngine


def test_pipeline_engine_full_run(mock_provider, sample_schema_plan, tmp_path):
    """Test the full pipeline with mocked providers."""
    plan_json = sample_schema_plan.model_dump_json()

    sample_batch = json.dumps([
        {
            "instruction": "Explain water's chemical formula.",
            "completion": "Water is H2O.",
            "difficulty": "easy",
        },
        {
            "instruction": "Explain photosynthesis in simple terms.",
            "completion": "Plants convert sunlight into stored chemical energy.",
            "difficulty": "medium",
        },
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
    assert ctx.plan.dataset_name == "instruction_pairs"
    assert len(ctx.validated_samples) == 2
    assert ctx.output_path is not None
    assert ctx.artifact_paths["artifacts_dir"].endswith("test_output_artifacts")
    assert Path(ctx.artifact_paths["accepted_jsonl"]).exists()
    assert Path(ctx.artifact_paths["rejected_jsonl"]).exists()
    assert Path(ctx.artifact_paths["metrics_json"]).exists()
    assert Path(ctx.artifact_paths["plan_json"]).exists()
    assert Path(ctx.artifact_paths["resolved_config_yaml"]).exists()
    assert Path(ctx.artifact_paths["report_md"]).exists()


def test_pipeline_engine_passes_per_agent_generation_config(
    mock_provider,
    sample_schema_plan,
    tmp_path,
):
    plan_json = sample_schema_plan.model_dump_json()
    sample_batch = json.dumps([
        {
            "instruction": "Explain H2O.",
            "completion": "Water is H2O.",
            "difficulty": "easy",
        },
    ])
    validation_result = json.dumps([
        {"index": 0, "is_valid": True, "score": 0.9, "issues": []},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider([validation_result])

    output_path = str(tmp_path / "test_output.jsonl")
    output_adapter = OutputAdapter.create("json", output_path)

    config = PipelineConfig(num_samples=1, batch_size=1)
    engine = PipelineEngine(
        planner,
        generator,
        validator,
        output_adapter,
        config,
        planner_generation_config=GenerationConfig(temperature=0.1, max_tokens=111),
        generator_generation_config=GenerationConfig(temperature=0.8, max_tokens=222),
        validator_generation_config=GenerationConfig(temperature=0.0, max_tokens=333),
    )
    engine.run("Generate chemistry Q&A", num_samples=1)

    assert planner.config_history[0] is not None
    assert planner.config_history[0].max_tokens == 111
    assert generator.config_history[0] is not None
    assert generator.config_history[0].max_tokens == 222
    assert validator.config_history[0] is not None
    assert validator.config_history[0].max_tokens == 333


def test_pipeline_engine_dedupes_exact_duplicates(mock_provider, sample_schema_plan, tmp_path):
    plan_json = sample_schema_plan.model_dump_json()
    duplicate_sample = {
        "instruction": "What is the chemical formula of water?",
        "completion": "Water is H2O.",
        "difficulty": "easy",
    }
    sample_batch = json.dumps([duplicate_sample, duplicate_sample])
    validation_result = json.dumps([
        {"index": 0, "is_valid": True, "score": 0.95, "issues": []},
        {"index": 1, "is_valid": True, "score": 0.92, "issues": []},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider([validation_result])

    output_path = str(tmp_path / "dedupe_output.jsonl")
    output_adapter = OutputAdapter.create("json", output_path)
    config = PipelineConfig(num_samples=2, batch_size=2)
    engine = PipelineEngine(planner, generator, validator, output_adapter, config)

    ctx = engine.run("Generate chemistry Q&A", num_samples=2)

    assert len(ctx.validated_samples) == 1
    assert ctx.metrics["duplicate_count"] == 1
    assert any("duplicate_exact" in item["issues"] for item in ctx.rejected_samples)


def test_pipeline_engine_optional_hf_postprocess(mock_provider, sample_schema_plan, tmp_path):
    plan_json = sample_schema_plan.model_dump_json()
    sample_batch = json.dumps([
        {"instruction": "Explain H2O.", "completion": "Water is H2O.", "difficulty": "easy"},
    ])
    validation_result = json.dumps([
        {"index": 0, "is_valid": True, "score": 0.99, "issues": []},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider([validation_result])

    output_path = str(tmp_path / "hf_post_source.jsonl")
    hf_output_path = str(tmp_path / "post_hf_dataset")
    output_adapter = OutputAdapter.create("json", output_path)
    config = PipelineConfig(
        num_samples=1,
        batch_size=1,
        postprocess_hf_from_jsonl=True,
        postprocess_hf_output_path=hf_output_path,
    )
    engine = PipelineEngine(planner, generator, validator, output_adapter, config)
    ctx = engine.run("Generate chemistry Q&A", num_samples=1)

    assert Path(ctx.artifact_paths["hf_dataset_path"]).exists()


def test_pipeline_engine_validation_failures_are_rejected(
    mock_provider,
    sample_schema_plan,
    tmp_path,
):
    plan_json = sample_schema_plan.model_dump_json()
    sample_batch = json.dumps([
        {"instruction": "Explain H2O.", "completion": "Water is H2O.", "difficulty": "easy"},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider(["not json"])

    output_path = str(tmp_path / "test_output.jsonl")
    output_adapter = OutputAdapter.create("json", output_path)
    config = PipelineConfig(num_samples=1, batch_size=1)
    engine = PipelineEngine(planner, generator, validator, output_adapter, config)

    ctx = engine.run("Generate chemistry Q&A", num_samples=1)

    assert len(ctx.validated_samples) == 0
    assert len(ctx.rejected_samples) == 1
    assert "validator_error:" in ctx.rejected_samples[0]["issues"][0]
    assert ctx.metrics["validation_error_batches"] == 1


def test_pipeline_engine_raises_on_generation_shortfall(
    mock_provider,
    sample_schema_plan,
    tmp_path,
):
    plan_json = sample_schema_plan.model_dump_json()
    sample_batch = json.dumps([
        {"instruction": "Explain H2O.", "completion": "Water is H2O.", "difficulty": "easy"},
    ])

    planner = mock_provider([plan_json])
    generator = mock_provider([sample_batch])
    validator = mock_provider([])

    output_path = str(tmp_path / "test_output.jsonl")
    output_adapter = OutputAdapter.create("json", output_path)
    config = PipelineConfig(num_samples=2, batch_size=2)
    engine = PipelineEngine(planner, generator, validator, output_adapter, config)

    with pytest.raises(GenerationError, match="wrong batch size"):
        engine.run("Generate chemistry Q&A", num_samples=2)
