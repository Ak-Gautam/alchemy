"""Tests for pipeline artifact writing."""

from __future__ import annotations

import json
from pathlib import Path

from alchemy.pipeline.artifacts import write_run_artifacts


def test_write_run_artifacts_for_jsonl_output(tmp_path):
    output_path = tmp_path / "dataset.jsonl"
    artifacts = write_run_artifacts(
        output_path=str(output_path),
        accepted_samples=[{"id": 1}],
        rejected_samples=[{"sample": {"id": 2}, "issues": ["bad"], "score": 0.1}],
        metrics={"valid_count": 1, "rejected_count": 1},
    )

    artifacts_dir = Path(artifacts["artifacts_dir"])
    assert artifacts_dir == tmp_path / "dataset_artifacts"
    assert Path(artifacts["accepted_jsonl"]).exists()
    assert Path(artifacts["rejected_jsonl"]).exists()
    assert Path(artifacts["metrics_json"]).exists()

    accepted_lines = Path(artifacts["accepted_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert len(accepted_lines) == 1
    assert json.loads(accepted_lines[0])["id"] == 1

    metrics = json.loads(Path(artifacts["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["valid_count"] == 1


def test_write_run_artifacts_for_directory_output(tmp_path):
    output_path = tmp_path / "hf_output"
    artifacts = write_run_artifacts(
        output_path=str(output_path),
        accepted_samples=[],
        rejected_samples=[],
        metrics={},
    )
    assert Path(artifacts["artifacts_dir"]) == output_path / "_artifacts"
