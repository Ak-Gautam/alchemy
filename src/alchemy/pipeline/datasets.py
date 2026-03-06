"""Dataset loading and merge helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alchemy.outputs.base import OutputAdapter
from alchemy.pipeline.plans import PlanType, validate_row_against_plan
from alchemy.quality.dedupe import dedupe_exact_rows
from alchemy.recipes import get_recipe


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load rows from a JSON/JSONL file, artifact directory, or HF dataset directory."""
    source = Path(path)
    if source.is_dir():
        accepted_jsonl = source / "accepted.jsonl"
        if accepted_jsonl.exists():
            return _load_jsonl_rows(accepted_jsonl)
        from datasets import Dataset

        dataset = Dataset.load_from_disk(str(source))
        return [dict(row) for row in dataset]

    if source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON array in {source}")
        return [dict(row) for row in payload]

    return _load_jsonl_rows(source)


def merge_rows(
    inputs: list[str | Path],
    *,
    plan: PlanType | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge inputs, returning `(accepted, duplicates, invalid)`."""
    rows: list[dict[str, Any]] = []
    for input_path in inputs:
        rows.extend(load_rows(input_path))

    invalid_rows: list[dict[str, Any]] = []
    valid_rows = rows
    if plan is not None:
        valid_rows = []
        for row in rows:
            issues = validate_row_against_plan(row, plan)
            issues.extend(_validate_recipe_rules(row, plan))
            if issues:
                invalid_rows.append({"sample": row, "issues": issues, "score": 0.0})
            else:
                valid_rows.append(row)

    unique_rows, duplicate_rows = dedupe_exact_rows(valid_rows)
    duplicate_rejections = [
        {"sample": row, "issues": ["duplicate_exact"], "score": 0.0}
        for row in duplicate_rows
    ]
    return unique_rows, duplicate_rejections, invalid_rows


def write_rows(
    *,
    rows: list[dict[str, Any]],
    output_path: str,
    output_format: str,
    plan: PlanType | None = None,
) -> str:
    if output_format.lower() not in {"json", "jsonl", "huggingface", "hf"}:
        raise ValueError("merge only supports json/jsonl and huggingface/hf outputs")

    adapter = OutputAdapter.create(output_format, output_path)
    return adapter.write(rows, plan)  # type: ignore[arg-type]


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _validate_recipe_rules(row: dict[str, Any], plan: PlanType) -> list[str]:
    metadata = getattr(plan, "metadata", {})
    if not isinstance(metadata, dict):
        return []
    recipe_name = metadata.get("recipe")
    if not isinstance(recipe_name, str):
        return []
    return get_recipe(recipe_name).validate_row_rules(row)
