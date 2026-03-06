"""Schema planner agent for JSON-schema-based generation plans."""

from __future__ import annotations

from typing import Any

from alchemy.models.base import GenerationResult
from alchemy.prompts.planner_prompts import build_schema_planner_system_prompt
from alchemy.spec.plan import GenerationPlan
from alchemy.utils.json_parsing import parse_json_payload

from .base import BaseAgent


class SchemaPlannerAgent(BaseAgent):
    """Analyzes a user prompt and produces a schema-first GenerationPlan."""

    expects_json = True

    def system_prompt(self, **kwargs: Any) -> str:
        return build_schema_planner_system_prompt(**kwargs)

    def parse_response(self, result: GenerationResult) -> GenerationPlan:
        data = parse_json_payload(result.text)
        return GenerationPlan.from_dict(data)

    def plan(self, user_prompt: str) -> GenerationPlan:
        """Create a schema-first generation plan from a user prompt."""
        return self.invoke(user_prompt)
