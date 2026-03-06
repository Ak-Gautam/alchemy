"""HuggingFace Dataset output adapter (default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import OutputAdapter


class HuggingFaceOutputAdapter(OutputAdapter):
    """Write samples as a HuggingFace Dataset saved to disk."""

    def write(self, samples: list[dict[str, Any]], plan: Any = None) -> str:
        from datasets import Dataset

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dataset = Dataset.from_list(samples)
        dataset.save_to_disk(str(path))

        return str(path)
