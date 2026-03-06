"""Pipeline run artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_run_artifacts(
    *,
    output_path: str,
    accepted_samples: list[dict[str, Any]],
    rejected_samples: list[dict[str, Any]],
    metrics: dict[str, Any],
    plan: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write run artifact files and return their paths.

    Core artifacts:
    - accepted.jsonl
    - rejected.jsonl
    - metrics.json

    Reporting artifacts:
    - plan.json (if provided)
    - resolved_config.yaml (if provided)
    - report.md
    """
    artifacts_dir = _resolve_artifacts_dir(Path(output_path))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = artifacts_dir / "accepted.jsonl"
    rejected_path = artifacts_dir / "rejected.jsonl"
    metrics_path = artifacts_dir / "metrics.json"
    plan_path = artifacts_dir / "plan.json"
    resolved_config_path = artifacts_dir / "resolved_config.yaml"
    report_path = artifacts_dir / "report.md"

    _write_jsonl(accepted_path, accepted_samples)
    _write_jsonl(rejected_path, rejected_samples)

    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, ensure_ascii=False)

    artifact_paths = {
        "artifacts_dir": str(artifacts_dir),
        "accepted_jsonl": str(accepted_path),
        "rejected_jsonl": str(rejected_path),
        "metrics_json": str(metrics_path),
    }
    if plan is not None:
        with plan_path.open("w", encoding="utf-8") as file_obj:
            json.dump(plan, file_obj, indent=2, ensure_ascii=False)
        artifact_paths["plan_json"] = str(plan_path)

    if resolved_config is not None:
        with resolved_config_path.open("w", encoding="utf-8") as file_obj:
            yaml.safe_dump(resolved_config, file_obj, sort_keys=False)
        artifact_paths["resolved_config_yaml"] = str(resolved_config_path)

    report_path.write_text(
        _build_report_markdown(
            output_path=output_path,
            accepted_count=len(accepted_samples),
            rejected_count=len(rejected_samples),
            metrics=metrics,
            artifact_paths=artifact_paths,
        ),
        encoding="utf-8",
    )
    artifact_paths["report_md"] = str(report_path)
    return artifact_paths


def _resolve_artifacts_dir(output_path: Path) -> Path:
    if output_path.suffix in {".json", ".jsonl"}:
        return output_path.parent / f"{output_path.stem}_artifacts"
    return output_path / "_artifacts"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_report_markdown(
    *,
    output_path: str,
    accepted_count: int,
    rejected_count: int,
    metrics: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    duplicate_count = metrics.get("duplicate_count", 0)
    planning_seconds = metrics.get("planning_seconds", "?")
    generation_seconds = metrics.get("generation_seconds", "?")
    validation_seconds = metrics.get("validation_seconds", "?")
    lines = [
        "# Run Report",
        "",
        "## Summary",
        f"- Output path: `{output_path}`",
        f"- Accepted samples: {accepted_count}",
        f"- Rejected samples: {rejected_count}",
        f"- Duplicates removed: {duplicate_count}",
        "",
        "## Timings",
        f"- Planning seconds: {planning_seconds}",
        f"- Generation seconds: {generation_seconds}",
        f"- Validation seconds: {validation_seconds}",
        "",
        "## Artifacts",
    ]
    for key in sorted(artifact_paths):
        lines.append(f"- {key}: `{artifact_paths[key]}`")
    return "\n".join(lines) + "\n"
