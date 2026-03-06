"""Tests for JSONL to HF postprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from alchemy.outputs.hf_postprocess import convert_jsonl_to_hf_dataset


def test_convert_jsonl_to_hf_dataset(tmp_path):
    jsonl_path = tmp_path / "accepted.jsonl"
    rows = [{"x": 1, "text": "a"}, {"x": 2, "text": "b"}]
    with jsonl_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row) + "\n")

    output_path = tmp_path / "hf_post"
    result_path = convert_jsonl_to_hf_dataset(str(jsonl_path), str(output_path))

    assert Path(result_path).exists()
    dataset = Dataset.load_from_disk(result_path)
    assert len(dataset) == 2
    assert dataset[0]["x"] == 1
