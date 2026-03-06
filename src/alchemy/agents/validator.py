"""Validator agent — checks generated samples for quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alchemy.models.base import GenerationResult
from alchemy.pipeline.plan import GenerationPlan
from alchemy.prompts.validator_prompts import build_validator_system_prompt

from .base import BaseAgent


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a single sample."""

    sample: dict[str, Any]
    is_valid: bool
    score: float
    issues: list[str]


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


class ValidatorAgent(BaseAgent):
    """Checks generated samples for quality and schema adherence."""

    def system_prompt(self, **kwargs: Any) -> str:
        plan: GenerationPlan = kwargs["plan"]
        return build_validator_system_prompt(plan)

    def parse_response(self, result: GenerationResult) -> list[dict[str, Any]]:
        return json.loads(_extract_json(result.text))

    def validate_batch(
        self,
        plan: GenerationPlan,
        samples: list[dict[str, Any]],
    ) -> list[ValidationResult]:
        """Validate a batch of samples against the plan."""
        user_msg = (
            f"Validate the following {len(samples)} samples against the schema.\n"
            f"For each sample, return: "
            f'{{"index": int, "is_valid": bool, "score": float (0-1), "issues": [str]}}.\n'
            f"Return a JSON array.\n\n"
            f"Samples:\n{json.dumps(samples, indent=2)}"
        )
        raw_results = self.invoke(user_msg, plan=plan)

        validated = []
        for i, vr in enumerate(raw_results):
            idx = vr.get("index", i)
            sample = samples[idx] if idx < len(samples) else samples[i]
            validated.append(
                ValidationResult(
                    sample=sample,
                    is_valid=vr["is_valid"],
                    score=vr.get("score", 1.0 if vr["is_valid"] else 0.0),
                    issues=vr.get("issues", []),
                )
            )
        return validated
