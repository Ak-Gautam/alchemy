"""Postprocessing helpers for converting JSONL outputs to HF datasets."""

from __future__ import annotations

import json
from pathlib import Path


def convert_jsonl_to_hf_dataset(jsonl_path: str, output_path: str) -> str:
    """Convert JSONL rows to a HuggingFace dataset saved to disk."""
    from datasets import Dataset

    source = Path(jsonl_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(destination))
    return str(destination)
