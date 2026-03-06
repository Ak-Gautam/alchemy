"""Tests for code output adapter."""

from __future__ import annotations

from pathlib import Path

from alchemy.outputs.code_output import CodeOutputAdapter
from alchemy.pipeline.plan import FieldSchema, GenerationPlan


def test_code_output_writes_files(tmp_path):
    plan = GenerationPlan(
        dataset_name="test_code",
        description="Test code generation",
        fields=[
            FieldSchema(
                name="code",
                field_type="code",
                description="Python code",
                constraints={"language": "python"},
            ),
            FieldSchema(
                name="description",
                field_type="string",
                description="What the code does",
            ),
        ],
        generation_strategy="random",
        num_samples=2,
    )

    path = str(tmp_path / "code_output")
    adapter = CodeOutputAdapter(path)
    samples = [
        {"code": "print('hello')", "description": "Prints hello"},
        {"code": "x = 1 + 2", "description": "Adds numbers"},
    ]
    result_path = adapter.write(samples, plan)
    out_dir = Path(result_path)
    assert out_dir.exists()
    assert (out_dir / "sample_00000.py").exists()
    assert (out_dir / "sample_00001.py").exists()
    assert (out_dir / "manifest.jsonl").exists()
