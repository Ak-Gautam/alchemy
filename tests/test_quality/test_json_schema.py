"""Tests for JSON schema structural validation."""

from __future__ import annotations

from alchemy.quality.json_schema import validate_row_against_schema, validate_rows_against_schema


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 5, "maxLength": 50},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "tags": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 2},
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "year": {"type": "integer", "minimum": 1900},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
        "required": ["text", "score", "difficulty"],
        "additionalProperties": False,
    }


def test_valid_row_has_no_issues():
    row = {
        "text": "Valid sample text",
        "score": 0.9,
        "difficulty": "easy",
        "tags": ["qa", "science"],
        "metadata": {"source": "synthetic", "year": 2024},
    }
    issues = validate_row_against_schema(row, _schema())
    assert issues == []


def test_required_and_additional_properties():
    row = {
        "text": "Valid sample text",
        "extra": "not allowed",
    }
    issues = validate_row_against_schema(row, _schema())
    assert any("missing required field 'score'" in issue for issue in issues)
    assert any("missing required field 'difficulty'" in issue for issue in issues)
    assert any("unexpected field 'extra'" in issue for issue in issues)


def test_type_and_enum_and_bounds_validation():
    row = {
        "text": "ok",
        "score": 2.0,
        "difficulty": "expert",
    }
    issues = validate_row_against_schema(row, _schema())
    assert any("string too short" in issue for issue in issues)
    assert any("above maximum" in issue for issue in issues)
    assert any("not in enum" in issue for issue in issues)


def test_nested_array_and_object_validation():
    row = {
        "text": "Valid enough text",
        "score": 0.5,
        "difficulty": "medium",
        "tags": ["x"],
        "metadata": {"year": 1800, "extra_nested": "nope"},
    }
    issues = validate_row_against_schema(row, _schema())
    assert any("$.tags[0]: string too short" in issue for issue in issues)
    assert any("$.metadata: missing required field 'source'" in issue for issue in issues)
    assert any("$.metadata.year: value 1800 below minimum 1900" in issue for issue in issues)
    assert any("$.metadata: unexpected field 'extra_nested'" in issue for issue in issues)


def test_validate_rows_returns_parallel_issue_lists():
    rows = [
        {"text": "short", "score": 0.1, "difficulty": "easy"},
        {"text": "bad", "score": "oops", "difficulty": "hard"},
    ]
    results = validate_rows_against_schema(rows, _schema())
    assert len(results) == 2
    assert results[0] == []
    assert any("expected number, got str" in issue for issue in results[1])
