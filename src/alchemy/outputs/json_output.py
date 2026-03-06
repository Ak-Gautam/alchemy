"""JSON / JSONL output adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alchemy.pipeline.plan import GenerationPlan

from .base import OutputAdapter


class JSONOutputAdapter(OutputAdapter):
    """Write samples as a JSONL file (one JSON object per line)."""

    def write(self, samples: list[dict[str, Any]], plan: GenerationPlan) -> str:
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default to .jsonl extension if none provided
        if path.suffix not in (".json", ".jsonl"):
            path = path.with_suffix(".jsonl")

        with open(path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        return str(path)
