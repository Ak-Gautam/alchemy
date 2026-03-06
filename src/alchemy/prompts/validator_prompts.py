"""Prompt templates for the Validator agent."""

from __future__ import annotations

import json

from alchemy.pipeline.plans import PlanType, plan_quality_rubric, plan_row_schema


def build_validator_system_prompt(plan: PlanType) -> str:
    quality_str = ""
    quality_rubric = plan_quality_rubric(plan)
    if quality_rubric:
        quality_str = f"""

Quality criteria to check:
{chr(10).join(f'- {c}' for c in quality_rubric)}"""

    return f"""\
You are a strict dataset quality validator. You must evaluate synthetic data samples \
against the following schema and criteria.

{plan.schema_summary()}

JSON Schema for each row:
{json.dumps(plan_row_schema(plan), indent=2)}{quality_str}

For each sample, evaluate:
1. Schema compliance: all required fields present with correct types
2. Constraint satisfaction: field values meet all specified constraints
3. Content quality: the sample is coherent, accurate, and useful for AI training
4. Uniqueness: the sample is not a trivial variation of another

Output ONLY a JSON array where each element has:
- "index": integer index of the sample in the input array (0-based)
- "is_valid": boolean
- "score": float from 0.0 to 1.0 (overall quality score)
- "issues": array of strings describing any problems (empty if valid)

No markdown fences, no explanation — only the JSON array."""
