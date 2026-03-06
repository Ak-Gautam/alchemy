"""Tests for structural validation."""

from __future__ import annotations

from alchemy.schemas.dynamic import validate_sample_structure


def test_valid_sample(sample_plan):
    sample = {
        "question": "What is the boiling point of water?",
        "answer": "100 degrees Celsius at standard pressure",
        "difficulty": "easy",
    }
    issues = validate_sample_structure(sample, sample_plan)
    assert issues == []


def test_missing_field(sample_plan):
    sample = {"question": "What is H2O?", "difficulty": "easy"}
    issues = validate_sample_structure(sample, sample_plan)
    assert any("Missing field: answer" in i for i in issues)


def test_field_too_short(sample_plan):
    sample = {"question": "Short?", "answer": "Yes", "difficulty": "easy"}
    issues = validate_sample_structure(sample, sample_plan)
    assert any("too short" in i for i in issues)


def test_extra_fields(sample_plan):
    sample = {
        "question": "What is the boiling point of water?",
        "answer": "100 degrees Celsius",
        "difficulty": "easy",
        "extra_field": "unexpected",
    }
    issues = validate_sample_structure(sample, sample_plan)
    assert any("Unexpected fields" in i for i in issues)


def test_wrong_type(sample_plan):
    sample = {
        "question": 12345,  # should be string
        "answer": "An answer here",
        "difficulty": "easy",
    }
    issues = validate_sample_structure(sample, sample_plan)
    assert any("expected string" in i for i in issues)
