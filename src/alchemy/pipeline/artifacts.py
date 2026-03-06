"""Pipeline run artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_artifacts(
    *,
    output_path: str,
    accepted_samples: list[dict[str, Any]],
    rejected_samples: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str]:
    """Write run artifact files and return their paths.

    Artifacts:
    - accepted.jsonl
    - rejected.jsonl
    - metrics.json
    """
    artifacts_dir = _resolve_artifacts_dir(Path(output_path))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = artifacts_dir / "accepted.jsonl"
    rejected_path = artifacts_dir / "rejected.jsonl"
    metrics_path = artifacts_dir / "metrics.json"

    _write_jsonl(accepted_path, accepted_samples)
    _write_jsonl(rejected_path, rejected_samples)

    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2, ensure_ascii=False)

    return {
        "artifacts_dir": str(artifacts_dir),
        "accepted_jsonl": str(accepted_path),
        "rejected_jsonl": str(rejected_path),
        "metrics_json": str(metrics_path),
    }


def _resolve_artifacts_dir(output_path: Path) -> Path:
    if output_path.suffix in {".json", ".jsonl"}:
        return output_path.parent / f"{output_path.stem}_artifacts"
    return output_path / "_artifacts"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
