"""Tests for exact deduplication utilities."""

from __future__ import annotations

from alchemy.quality.dedupe import dedupe_exact_rows


def test_dedupe_exact_rows_removes_duplicates():
    rows = [
        {"a": 1, "b": "x"},
        {"b": "x", "a": 1},  # same content, different key order
        {"a": 2, "b": "y"},
    ]
    unique_rows, duplicate_rows = dedupe_exact_rows(rows)
    assert len(unique_rows) == 2
    assert len(duplicate_rows) == 1
    assert unique_rows[0] == {"a": 1, "b": "x"}
    assert unique_rows[1] == {"a": 2, "b": "y"}
