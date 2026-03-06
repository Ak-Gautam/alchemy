"""Planner agent — designs the dataset schema and generation strategy."""

from __future__ import annotations

from typing import Any

from alchemy.models.base import GenerationResult
from alchemy.pipeline.plan import GenerationPlan
from alchemy.prompts.planner_prompts import build_planner_system_prompt
from alchemy.utils.json_parsing import parse_json_payload

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    """Analyzes a user prompt and produces a GenerationPlan."""

    expects_json = True

    def system_prompt(self, **kwargs: Any) -> str:
        return build_planner_system_prompt(**kwargs)

    def parse_response(self, result: GenerationResult) -> GenerationPlan:
        data = parse_json_payload(result.text)
        return GenerationPlan.from_dict(data)

    def plan(self, user_prompt: str) -> GenerationPlan:
        """Create a generation plan from a user prompt."""
        return self.invoke(user_prompt)
