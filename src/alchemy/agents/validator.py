"""Validator agent — checks generated samples for quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alchemy.exceptions import ValidationError
from alchemy.models.base import GenerationResult
from alchemy.prompts.validator_prompts import build_validator_system_prompt
from alchemy.utils.json_parsing import parse_json_payload

from .base import BaseAgent


@dataclass(slots=True)
class ValidationResult:
    """Result of validating a single sample."""

    sample: dict[str, Any]
    is_valid: bool
    score: float
    issues: list[str]

class ValidatorAgent(BaseAgent):
    """Checks generated samples for quality and schema adherence."""

    expects_json = True

    def system_prompt(self, **kwargs: Any) -> str:
        plan = kwargs["plan"]
        return build_validator_system_prompt(plan)

    def parse_response(self, result: GenerationResult) -> list[dict[str, Any]]:
        data = parse_json_payload(result.text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of validation results, got {type(data).__name__}")
        return data

    def validate_batch(
        self,
        plan: Any,
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

        if len(raw_results) != len(samples):
            raise ValidationError(
                "Validator returned the wrong number of results",
                details=f"expected {len(samples)}, got {len(raw_results)}",
            )

        validated = []
        seen_indices: set[int] = set()
        for i, vr in enumerate(raw_results):
            idx = vr.get("index", i)
            if not isinstance(idx, int) or idx < 0 or idx >= len(samples):
                raise ValidationError(
                    "Validator returned an invalid sample index",
                    details=f"index={idx!r}, batch_size={len(samples)}",
                )
            if idx in seen_indices:
                raise ValidationError(
                    "Validator returned duplicate sample indices",
                    details=f"duplicate index {idx}",
                )
            seen_indices.add(idx)
            sample = samples[idx]
            validated.append(
                ValidationResult(
                    sample=sample,
                    is_valid=vr["is_valid"],
                    score=vr.get("score", 1.0 if vr["is_valid"] else 0.0),
                    issues=vr.get("issues", []),
                )
            )
        return validated
