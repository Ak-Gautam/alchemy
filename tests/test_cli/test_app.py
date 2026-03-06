"""Tests for CLI app."""

from __future__ import annotations

from typer.testing import CliRunner

from alchemy.cli.app import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Synthetic data generation" in result.output


def test_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "num-samples" in result.output
    assert "format" in result.output


def test_plan_help():
    result = runner.invoke(app, ["plan", "--help"])
    assert result.exit_code == 0


def test_validate_help():
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
