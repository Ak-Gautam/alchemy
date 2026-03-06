"""Lightweight structural validation against a JSON Schema subset."""

from __future__ import annotations

from typing import Any


_TYPE_NAME_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def validate_row_against_schema(row: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate one row against a JSON Schema subset.

    Supported keywords:
    - type
    - properties
    - required
    - additionalProperties (bool)
    - enum
    - minLength / maxLength
    - minimum / maximum
    - minItems / maxItems
    - items
    """
    issues: list[str] = []
    _validate_value(value=row, schema=schema, path="$", issues=issues)
    return issues


def validate_rows_against_schema(
    rows: list[dict[str, Any]],
    schema: dict[str, Any],
) -> list[list[str]]:
    """Validate multiple rows and return per-row issue lists."""
    return [validate_row_against_schema(row, schema) for row in rows]


def _validate_value(
    *,
    value: Any,
    schema: dict[str, Any],
    path: str,
    issues: list[str],
) -> None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        expected_types = _TYPE_NAME_MAP.get(schema_type)
        if expected_types is None:
            issues.append(f"{path}: unsupported schema type '{schema_type}'")
            return

        if not isinstance(value, expected_types) or (
            schema_type == "integer" and isinstance(value, bool)
        ):
            issues.append(f"{path}: expected {schema_type}, got {type(value).__name__}")
            return

    if "enum" in schema:
        enum_values = schema.get("enum", [])
        if value not in enum_values:
            issues.append(f"{path}: value {value!r} not in enum {enum_values!r}")

    if isinstance(value, str):
        _validate_string(value=value, schema=schema, path=path, issues=issues)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(value=value, schema=schema, path=path, issues=issues)
    elif isinstance(value, list):
        _validate_array(value=value, schema=schema, path=path, issues=issues)
    elif isinstance(value, dict):
        _validate_object(value=value, schema=schema, path=path, issues=issues)


def _validate_string(
    *,
    value: str,
    schema: dict[str, Any],
    path: str,
    issues: list[str],
) -> None:
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        issues.append(f"{path}: string too short ({len(value)} < {min_length})")

    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        issues.append(f"{path}: string too long ({len(value)} > {max_length})")


def _validate_number(
    *,
    value: int | float,
    schema: dict[str, Any],
    path: str,
    issues: list[str],
) -> None:
    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and value < minimum:
        issues.append(f"{path}: value {value} below minimum {minimum}")

    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and value > maximum:
        issues.append(f"{path}: value {value} above maximum {maximum}")


def _validate_array(
    *,
    value: list[Any],
    schema: dict[str, Any],
    path: str,
    issues: list[str],
) -> None:
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        issues.append(f"{path}: array too short ({len(value)} < {min_items})")

    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        issues.append(f"{path}: array too long ({len(value)} > {max_items})")

    items_schema = schema.get("items")
    if isinstance(items_schema, dict):
        for index, item in enumerate(value):
            _validate_value(
                value=item,
                schema=items_schema,
                path=f"{path}[{index}]",
                issues=issues,
            )


def _validate_object(
    *,
    value: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    issues: list[str],
) -> None:
    required = schema.get("required", [])
    if isinstance(required, list):
        for field_name in required:
            if field_name not in value:
                issues.append(f"{path}: missing required field '{field_name}'")

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for field_name, field_schema in properties.items():
            if field_name not in value:
                continue
            if not isinstance(field_schema, dict):
                continue
            _validate_value(
                value=value[field_name],
                schema=field_schema,
                path=f"{path}.{field_name}",
                issues=issues,
            )

    additional_properties = schema.get("additionalProperties")
    if additional_properties is False and isinstance(properties, dict):
        extra_fields = sorted(set(value) - set(properties))
        for field_name in extra_fields:
            issues.append(f"{path}: unexpected field '{field_name}'")
