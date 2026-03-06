"""Tests for HuggingFace output adapter."""

from __future__ import annotations

from pathlib import Path

from alchemy.outputs.hf_dataset import HuggingFaceOutputAdapter


def test_hf_output_saves_dataset(tmp_path, sample_plan):
    path = str(tmp_path / "hf_output")
    adapter = HuggingFaceOutputAdapter(path)
    samples = [
        {"question": "Q1?", "answer": "A1", "difficulty": "easy"},
        {"question": "Q2?", "answer": "A2", "difficulty": "hard"},
    ]
    result_path = adapter.write(samples, sample_plan)
    assert Path(result_path).exists()

    # Verify we can load it back
    from datasets import Dataset
    ds = Dataset.load_from_disk(result_path)
    assert len(ds) == 2
    assert ds[0]["question"] == "Q1?"
