"""Output adapter interface."""

from __future__ import annotations

import abc
from typing import Any

from alchemy.pipeline.plans import PlanType


class OutputAdapter(abc.ABC):
    """Interface for writing validated samples to a specific format."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    @abc.abstractmethod
    def write(self, samples: list[dict[str, Any]], plan: PlanType | None) -> str:
        """Write samples and return the output path."""
        ...

    @staticmethod
    def create(format_name: str, output_path: str) -> OutputAdapter:
        from .code_output import CodeOutputAdapter
        from .hf_dataset import HuggingFaceOutputAdapter
        from .json_output import JSONOutputAdapter

        adapters: dict[str, type[OutputAdapter]] = {
            "json": JSONOutputAdapter,
            "jsonl": JSONOutputAdapter,
            "code": CodeOutputAdapter,
            "huggingface": HuggingFaceOutputAdapter,
            "hf": HuggingFaceOutputAdapter,
        }
        cls = adapters.get(format_name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown output format: {format_name!r}. "
                f"Available: {list(adapters.keys())}"
            )
        return cls(output_path)
