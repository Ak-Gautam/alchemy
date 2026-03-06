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
        plan={"dataset_name": "demo"},
        resolved_config={"output_format": "json"},
    )

    artifacts_dir = Path(artifacts["artifacts_dir"])
    assert artifacts_dir == tmp_path / "dataset_artifacts"
    assert Path(artifacts["accepted_jsonl"]).exists()
    assert Path(artifacts["rejected_jsonl"]).exists()
    assert Path(artifacts["metrics_json"]).exists()
    assert Path(artifacts["plan_json"]).exists()
    assert Path(artifacts["resolved_config_yaml"]).exists()
    assert Path(artifacts["report_md"]).exists()

    accepted_lines = Path(artifacts["accepted_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert len(accepted_lines) == 1
    assert json.loads(accepted_lines[0])["id"] == 1

    metrics = json.loads(Path(artifacts["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["valid_count"] == 1
    report_text = Path(artifacts["report_md"]).read_text(encoding="utf-8")
    assert "Run Report" in report_text
    assert "Accepted samples: 1" in report_text


def test_write_run_artifacts_for_directory_output(tmp_path):
    output_path = tmp_path / "hf_output"
    artifacts = write_run_artifacts(
        output_path=str(output_path),
        accepted_samples=[],
        rejected_samples=[],
        metrics={},
    )
    assert Path(artifacts["artifacts_dir"]) == output_path / "_artifacts"
    assert Path(artifacts["report_md"]).exists()
