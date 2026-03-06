"""Fast structural validation without LLM calls."""

from __future__ import annotations

from typing import Any

from alchemy.pipeline.plan import GenerationPlan

# Maps field_type strings to expected Python types
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "float": (int, float),
    "boolean": bool,
    "code": str,
    "json": (dict, list),
}


def validate_sample_structure(
    sample: dict[str, Any], plan: GenerationPlan
) -> list[str]:
    """Structurally validate a sample against the plan's field schema.

    Returns a list of issues (empty means valid).
    """
    issues: list[str] = []

    for field_def in plan.fields:
        if field_def.name not in sample:
            issues.append(f"Missing field: {field_def.name}")
            continue

        value = sample[field_def.name]

        # Type checking
        if field_def.field_type.startswith("list["):
            if not isinstance(value, list):
                issues.append(
                    f"Field {field_def.name}: expected list, got {type(value).__name__}"
                )
        elif field_def.field_type in _TYPE_MAP:
            expected = _TYPE_MAP[field_def.field_type]
            if not isinstance(value, expected):
                issues.append(
                    f"Field {field_def.name}: expected {field_def.field_type}, "
                    f"got {type(value).__name__}"
                )

        # Constraint checking for strings
        constraints = field_def.constraints
        if isinstance(value, str):
            if "min_length" in constraints and len(value) < constraints["min_length"]:
                issues.append(
                    f"Field {field_def.name}: too short "
                    f"({len(value)} < {constraints['min_length']})"
                )
            if "max_length" in constraints and len(value) > constraints["max_length"]:
                issues.append(
                    f"Field {field_def.name}: too long "
                    f"({len(value)} > {constraints['max_length']})"
                )

        # Constraint checking for numbers
        if isinstance(value, (int, float)):
            if "min_value" in constraints and value < constraints["min_value"]:
                issues.append(
                    f"Field {field_def.name}: value {value} below minimum "
                    f"{constraints['min_value']}"
                )
            if "max_value" in constraints and value > constraints["max_value"]:
                issues.append(
                    f"Field {field_def.name}: value {value} above maximum "
                    f"{constraints['max_value']}"
                )

    # Check for unexpected extra fields
    expected_names = {f.name for f in plan.fields}
    extra = set(sample.keys()) - expected_names
    if extra:
        issues.append(f"Unexpected fields: {', '.join(sorted(extra))}")

    return issues
