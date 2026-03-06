"""Helpers for working with both legacy and schema-first plans."""

from __future__ import annotations

from typing import Any

from alchemy.pipeline.plan import GenerationPlan as LegacyGenerationPlan
from alchemy.quality.json_schema import validate_row_against_schema
from alchemy.spec.plan import GenerationPlan as SchemaGenerationPlan

PlanType = LegacyGenerationPlan | SchemaGenerationPlan


def is_schema_plan(plan: PlanType) -> bool:
    return isinstance(plan, SchemaGenerationPlan)


def load_plan(data: dict[str, Any]) -> PlanType:
    """Load a plan JSON payload as schema-first or legacy."""
    try:
        return SchemaGenerationPlan.from_dict(data)
    except Exception:
        return LegacyGenerationPlan.from_dict(data)


def plan_total_rows(plan: PlanType) -> int:
    if is_schema_plan(plan):
        return plan.defaults.num_rows
    return plan.num_samples


def plan_batch_size(plan: PlanType) -> int:
    if is_schema_plan(plan):
        return min(plan.defaults.batch_size, plan.defaults.num_rows)
    return min(plan.batch_size, plan.num_samples)


def set_plan_total_rows(plan: PlanType, total_rows: int) -> None:
    if is_schema_plan(plan):
        plan.defaults.num_rows = total_rows
        if plan.defaults.batch_size > total_rows:
            plan.defaults.batch_size = total_rows
        return
    plan.num_samples = total_rows
    if plan.batch_size > total_rows:
        plan.batch_size = total_rows


def plan_row_schema(plan: PlanType) -> dict[str, Any]:
    if is_schema_plan(plan):
        return plan.row_schema
    return _legacy_plan_to_schema(plan)


def plan_example_rows(plan: PlanType) -> list[dict[str, Any]]:
    if is_schema_plan(plan):
        return plan.example_rows
    return plan.example_samples


def plan_quality_rubric(plan: PlanType) -> list[str]:
    if is_schema_plan(plan):
        return plan.quality_rubric
    return plan.quality_criteria


def plan_variation_axes(plan: PlanType) -> list[str]:
    if is_schema_plan(plan):
        return [axis.name for axis in plan.variation_spec.axes]
    return plan.diversity_dimensions


def plan_field_names(plan: PlanType) -> list[str]:
    if is_schema_plan(plan):
        properties = plan.row_schema.get("properties", {})
        return sorted(properties.keys())
    return [field.name for field in plan.fields]


def plan_generation_strategy(plan: PlanType) -> str:
    if is_schema_plan(plan):
        axes = plan_variation_axes(plan)
        if axes:
            return f"Cover variation across: {', '.join(axes)}."
        return "Generate diverse rows that satisfy the schema and quality rubric."
    return plan.generation_strategy


def plan_code_field(plan: PlanType) -> tuple[str | None, str]:
    if is_schema_plan(plan):
        properties = plan.row_schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue
            if field_schema.get("contentMediaType") == "text/x-code":
                return field_name, _language_to_ext(field_schema.get("x-language", ""))
        return None, ".txt"

    for field in plan.fields:
        if field.field_type != "code":
            continue
        language = ""
        if isinstance(field.constraints, dict):
            language = str(field.constraints.get("language", ""))
        return field.name, _language_to_ext(language)
    return None, ".txt"


def validate_row_against_plan(row: dict[str, Any], plan: PlanType) -> list[str]:
    return validate_row_against_schema(row, plan_row_schema(plan))


def _legacy_plan_to_schema(plan: LegacyGenerationPlan) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in plan.fields:
        field_schema: dict[str, Any] = _legacy_field_schema(field.field_type)
        if field.description:
            field_schema["description"] = field.description
        _apply_legacy_constraints(field_schema, field.constraints)
        properties[field.name] = field_schema
        required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _legacy_field_schema(field_type: str) -> dict[str, Any]:
    if field_type == "string":
        return {"type": "string"}
    if field_type == "integer":
        return {"type": "integer"}
    if field_type == "float":
        return {"type": "number"}
    if field_type == "boolean":
        return {"type": "boolean"}
    if field_type == "json":
        return {"type": "object"}
    if field_type == "code":
        return {"type": "string", "contentMediaType": "text/x-code"}
    if field_type.startswith("list[") and field_type.endswith("]"):
        inner_type = field_type[5:-1]
        return {"type": "array", "items": _legacy_field_schema(inner_type)}
    return {"type": "string"}


def _apply_legacy_constraints(field_schema: dict[str, Any], constraints: dict[str, Any]) -> None:
    if "min_length" in constraints:
        field_schema["minLength"] = constraints["min_length"]
    if "max_length" in constraints:
        field_schema["maxLength"] = constraints["max_length"]
    if "min_value" in constraints:
        field_schema["minimum"] = constraints["min_value"]
    if "max_value" in constraints:
        field_schema["maximum"] = constraints["max_value"]
    if "enum" in constraints:
        field_schema["enum"] = constraints["enum"]
    if field_schema.get("contentMediaType") == "text/x-code" and "language" in constraints:
        field_schema["x-language"] = constraints["language"]


def _language_to_ext(language: str) -> str:
    mapping = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "ruby": ".rb",
        "swift": ".swift",
        "kotlin": ".kt",
        "sql": ".sql",
        "html": ".html",
        "css": ".css",
        "shell": ".sh",
        "bash": ".sh",
    }
    return mapping.get(language.lower(), ".txt")
