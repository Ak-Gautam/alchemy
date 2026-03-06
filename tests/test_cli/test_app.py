"""Tests for CLI app."""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from alchemy.cli import app as cli_module
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


def test_recipes_command_lists_builtin_recipe():
    result = runner.invoke(app, ["recipes"])
    assert result.exit_code == 0
    assert "text_compression_pairs" in result.output


def test_validate_requires_prompt_or_plan_file(tmp_path):
    input_path = tmp_path / "data.jsonl"
    input_path.write_text('{"x":1}\n', encoding="utf-8")
    result = runner.invoke(app, ["validate", str(input_path)])
    assert result.exit_code == 1
    assert "provide either --plan-file or --prompt" in result.output


def test_generate_sets_hf_postprocess_flags(monkeypatch, tmp_path):
    captured = {}

    class _FakeEngine:
        def run(self, prompt: str, num_samples: int | None = None):
            captured["prompt"] = prompt
            captured["num_samples"] = num_samples
            return SimpleNamespace(
                metrics={"valid_count": 1},
                artifact_paths={"artifacts_dir": str(tmp_path / "artifacts")},
            )

    def _fake_from_config(config):
        captured["config"] = config
        return _FakeEngine()

    monkeypatch.setattr(cli_module.PipelineEngine, "from_config", staticmethod(_fake_from_config))

    result = runner.invoke(
        app,
        [
            "generate",
            "make data",
            "--recipe",
            "text_compression_pairs",
            "--postprocess-hf-from-jsonl",
            "--postprocess-hf-output",
            str(tmp_path / "hf_dataset"),
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.postprocess_hf_from_jsonl is True
    assert config.postprocess_hf_output_path == str(tmp_path / "hf_dataset")
    assert captured["num_samples"] == 1000
    assert "Recipe constraints:" in captured["prompt"]
