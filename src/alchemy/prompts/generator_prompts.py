"""Prompt templates for the Generator agent."""

from __future__ import annotations

import json

from alchemy.pipeline.plans import (
    PlanType,
    plan_example_rows,
    plan_field_names,
    plan_generation_strategy,
    plan_quality_rubric,
    plan_row_schema,
    plan_variation_axes,
)


def build_generator_system_prompt(plan: PlanType) -> str:
    examples_str = ""
    example_rows = plan_example_rows(plan)
    if example_rows:
        examples_str = f"""

Here are example samples for reference:
{json.dumps(example_rows, indent=2)}"""

    diversity_str = ""
    variation_axes = plan_variation_axes(plan)
    if variation_axes:
        diversity_str = f"""

Diversity dimensions to vary across batches: {', '.join(variation_axes)}"""

    quality_str = ""
    quality_rubric = plan_quality_rubric(plan)
    if quality_rubric:
        quality_str = f"""

Quality rubric:
{chr(10).join(f'- {item}' for item in quality_rubric)}"""

    return f"""\
You are a high-quality synthetic data generator. Your task is to generate samples \
for the following dataset:

{plan.schema_summary()}

JSON Schema for each row:
{json.dumps(plan_row_schema(plan), indent=2)}

Generation strategy: {plan_generation_strategy(plan)}{examples_str}{diversity_str}{quality_str}

RULES:
1. Output ONLY a JSON array of sample objects — no markdown fences, no explanation.
2. Every sample MUST contain exactly these fields with correct types: \
{', '.join(plan_field_names(plan))}.
3. Each sample must be unique, diverse, and high quality.
4. Follow all field constraints strictly.
5. Vary content across the requested diversity dimensions.
6. Do NOT repeat or closely paraphrase the example samples."""
