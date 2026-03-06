"""Strict language constraint checks for code-generation datasets."""

from __future__ import annotations

from typing import Any


_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "python": ("def ", "import ", "class ", "lambda ", "if __name__"),
    "rust": ("fn ", "let ", "impl ", "use ", "::"),
    "javascript": ("function ", "const ", "let ", "=>", "import "),
    "typescript": ("function ", "const ", "let ", "interface ", "type "),
}


def validate_language_constraints(
    sample: dict[str, Any],
    *,
    code_field: str = "code",
    language: str,
    disallowed_substrings: list[str] | None = None,
    required_substrings: list[str] | None = None,
) -> list[str]:
    """Validate language and policy constraints for code content."""
    issues: list[str] = []
    value = sample.get(code_field)
    if not isinstance(value, str):
        return [f"Missing or invalid code field: {code_field}"]

    code = value.strip()
    if not code:
        issues.append("Code content is empty")
        return issues

    normalized_language = language.lower()
    markers = _LANGUAGE_MARKERS.get(normalized_language, ())
    if markers and not any(marker in code for marker in markers):
        issues.append(
            f"Code does not appear to match declared language '{normalized_language}'"
        )

    for substring in disallowed_substrings or []:
        if substring in code:
            issues.append(f"Disallowed substring found: {substring}")

    for substring in required_substrings or []:
        if substring not in code:
            issues.append(f"Required substring missing: {substring}")

    return issues
