"""Planner agent — designs the dataset schema and generation strategy."""

from __future__ import annotations

import json
from typing import Any

from alchemy.models.base import GenerationResult
from alchemy.pipeline.plan import GenerationPlan
from alchemy.prompts.planner_prompts import build_planner_system_prompt

from .base import BaseAgent


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


class PlannerAgent(BaseAgent):
    """Analyzes a user prompt and produces a GenerationPlan."""

    def system_prompt(self, **kwargs: Any) -> str:
        return build_planner_system_prompt(**kwargs)

    def parse_response(self, result: GenerationResult) -> GenerationPlan:
        data = json.loads(_extract_json(result.text))
        return GenerationPlan.from_dict(data)

    def plan(self, user_prompt: str) -> GenerationPlan:
        """Create a generation plan from a user prompt."""
        return self.invoke(user_prompt)
