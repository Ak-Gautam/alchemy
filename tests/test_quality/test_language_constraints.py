"""Tests for strict language constraint checks."""

from __future__ import annotations

from alchemy.quality.language_constraints import validate_language_constraints
from alchemy.recipes.base import BaseRecipe, LanguageConstraints


class _PythonCodeRecipe(BaseRecipe):
    @property
    def name(self) -> str:
        return "python_code_recipe"

    @property
    def description(self) -> str:
        return "Test recipe for language constraints."

    @property
    def row_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }

    @property
    def language_constraints(self) -> LanguageConstraints:
        return LanguageConstraints(
            language="python",
            code_field="code",
            disallowed_substrings=["os.system("],
            required_substrings=["def "],
        )


def test_validate_language_constraints_detects_mismatch_and_policy_violations():
    issues = validate_language_constraints(
        {"code": "fn main() { println!(\"hi\"); }"},
        language="python",
        disallowed_substrings=["println!"],
        required_substrings=["def "],
    )
    assert any("does not appear to match declared language 'python'" in issue for issue in issues)
    assert any("Disallowed substring found: println!" in issue for issue in issues)
    assert any("Required substring missing: def " in issue for issue in issues)


def test_validate_language_constraints_passes_for_valid_python():
    issues = validate_language_constraints(
        {"code": "def add(a, b):\n    return a + b"},
        language="python",
        required_substrings=["def "],
    )
    assert issues == []


def test_recipe_validate_row_rules_uses_language_constraints():
    recipe = _PythonCodeRecipe()
    issues = recipe.validate_row_rules({"code": "print('hello')"})
    assert any("Required substring missing: def " in issue for issue in issues)
