"""Exact deduplication utilities for generated samples."""

from __future__ import annotations

import json
from typing import Any


def dedupe_exact_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate rows by canonical JSON representation.

    Returns `(unique_rows, duplicate_rows)` preserving the first occurrence order.
    """
    seen_hashes: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    for row in rows:
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if canonical in seen_hashes:
            duplicate_rows.append(row)
            continue
        seen_hashes.add(canonical)
        unique_rows.append(row)

    return unique_rows, duplicate_rows
