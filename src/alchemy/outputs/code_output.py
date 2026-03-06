"""Code file output adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alchemy.pipeline.plans import PlanType, plan_code_field

from .base import OutputAdapter


class CodeOutputAdapter(OutputAdapter):
    """Write samples as individual code files.

    Looks for a field with field_type "code" to extract the code content.
    Falls back to writing each sample as a JSON file.
    """

    def write(self, samples: list[dict[str, Any]], plan: PlanType | None) -> str:
        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the code field and optional language constraint
        code_field, extension = (None, ".txt")
        if plan is not None:
            code_field, extension = plan_code_field(plan)

        for i, sample in enumerate(samples):
            filename = f"sample_{i:05d}"
            if code_field and code_field in sample:
                filepath = out_dir / f"{filename}{extension}"
                filepath.write_text(sample[code_field], encoding="utf-8")
            else:
                filepath = out_dir / f"{filename}.json"
                filepath.write_text(
                    json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        # Also write a manifest
        manifest = out_dir / "manifest.jsonl"
        with open(manifest, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        return str(out_dir)
