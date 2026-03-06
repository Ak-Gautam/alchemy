"""Generator agent — produces synthetic data samples in batches."""

from __future__ import annotations

from typing import Any

from alchemy.models.base import GenerationResult
from alchemy.pipeline.plan import GenerationPlan
from alchemy.prompts.generator_prompts import build_generator_system_prompt
from alchemy.utils.json_parsing import parse_json_payload

from .base import BaseAgent


class GeneratorAgent(BaseAgent):
    """Produces synthetic data samples according to a GenerationPlan."""

    def system_prompt(self, **kwargs: Any) -> str:
        plan: GenerationPlan = kwargs["plan"]
        return build_generator_system_prompt(plan)

    def parse_response(self, result: GenerationResult) -> list[dict[str, Any]]:
        data = parse_json_payload(result.text)
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a list of samples, got {type(data).__name__}")

    def generate_batch(
        self,
        plan: GenerationPlan,
        batch_size: int = 10,
        batch_index: int = 0,
    ) -> list[dict[str, Any]]:
        """Generate a single batch of samples."""
        user_msg = (
            f"Generate batch {batch_index + 1} with exactly {batch_size} unique samples. "
            f"Return a JSON array of {batch_size} sample objects."
        )
        return self.invoke(user_msg, plan=plan)
