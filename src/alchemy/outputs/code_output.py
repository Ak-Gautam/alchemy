"""Code file output adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alchemy.pipeline.plan import GenerationPlan

from .base import OutputAdapter


class CodeOutputAdapter(OutputAdapter):
    """Write samples as individual code files.

    Looks for a field with field_type "code" to extract the code content.
    Falls back to writing each sample as a JSON file.
    """

    def write(self, samples: list[dict[str, Any]], plan: GenerationPlan) -> str:
        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Find the code field and optional language constraint
        code_field = None
        extension = ".txt"
        for field_def in plan.fields:
            if field_def.field_type == "code":
                code_field = field_def.name
                lang = field_def.constraints.get("language", "")
                extension = _lang_to_ext(lang)
                break

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


def _lang_to_ext(language: str) -> str:
    """Map common language names to file extensions."""
    mapping = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "ruby": ".rb",
        "swift": ".swift",
        "kotlin": ".kt",
        "sql": ".sql",
        "html": ".html",
        "css": ".css",
        "shell": ".sh",
        "bash": ".sh",
    }
    return mapping.get(language.lower(), ".txt")
