"""Prompt templates for the Generator agent."""

from __future__ import annotations

import json

from alchemy.pipeline.plan import GenerationPlan


def build_generator_system_prompt(plan: GenerationPlan) -> str:
    examples_str = ""
    if plan.example_samples:
        examples_str = f"""

Here are example samples for reference:
{json.dumps(plan.example_samples, indent=2)}"""

    diversity_str = ""
    if plan.diversity_dimensions:
        diversity_str = f"""

Diversity dimensions to vary across batches: {', '.join(plan.diversity_dimensions)}"""

    return f"""\
You are a high-quality synthetic data generator. Your task is to generate samples \
for the following dataset:

{plan.schema_summary()}

Generation strategy: {plan.generation_strategy}{examples_str}{diversity_str}

RULES:
1. Output ONLY a JSON array of sample objects — no markdown fences, no explanation.
2. Every sample MUST contain exactly these fields with correct types: \
{', '.join(f.name for f in plan.fields)}.
3. Each sample must be unique, diverse, and high quality.
4. Follow all field constraints strictly.
5. Vary content across the requested diversity dimensions.
6. Do NOT repeat or closely paraphrase the example samples."""
