"""Tests for CLI app."""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from alchemy.cli import app as cli_module
from alchemy.cli.app import app
from alchemy.recipes import get_recipe

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
        def run(
            self,
            prompt: str,
            num_samples: int | None = None,
            *,
            plan=None,
            chunk_id: str | None = None,
        ):
            captured["prompt"] = prompt
            captured["num_samples"] = num_samples
            captured["plan"] = plan
            captured["chunk_id"] = chunk_id
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
    assert captured["plan"].dataset_name == get_recipe("text_compression_pairs").name


def test_generate_reuses_saved_plan(monkeypatch, tmp_path, sample_schema_plan):
    captured = {}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(sample_schema_plan.model_dump_json(), encoding="utf-8")

    class _FakeEngine:
        def run(
            self,
            prompt: str,
            num_samples: int | None = None,
            *,
            plan=None,
            chunk_id: str | None = None,
        ):
            captured["plan"] = plan
            captured["chunk_id"] = chunk_id
            return SimpleNamespace(metrics={"valid_count": 1}, artifact_paths={})

    def _fake_from_config(config):
        return _FakeEngine()

    monkeypatch.setattr(cli_module.PipelineEngine, "from_config", staticmethod(_fake_from_config))

    result = runner.invoke(
        app,
        [
            "generate",
            "make data",
            "--plan-file",
            str(plan_path),
            "--num-samples",
            "25",
            "--chunk-id",
            "chunk-03",
        ],
    )

    assert result.exit_code == 0
    assert captured["plan"].defaults.num_rows == 25
    assert captured["chunk_id"] == "chunk-03"


def test_plan_command_can_save_json(monkeypatch, tmp_path, sample_schema_plan):
    class _FakeEngine:
        def run_plan_only(self, prompt: str, *, plan=None):
            return plan or sample_schema_plan

    def _fake_from_config(config):
        return _FakeEngine()

    monkeypatch.setattr(cli_module.PipelineEngine, "from_config", staticmethod(_fake_from_config))

    output_path = tmp_path / "plan.json"
    result = runner.invoke(app, ["plan", "make data", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "instruction_pairs" in output_path.read_text(encoding="utf-8")


def test_merge_command_merges_chunks(tmp_path):
    chunk1 = tmp_path / "chunk1.jsonl"
    chunk1.write_text(
        '{"instruction":"a","completion":"A","difficulty":"easy"}\n'
        '{"instruction":"b","completion":"B","difficulty":"medium"}\n',
        encoding="utf-8",
    )
    chunk2 = tmp_path / "chunk2.jsonl"
    chunk2.write_text(
        '{"instruction":"b","completion":"B","difficulty":"medium"}\n'
        '{"instruction":"c","completion":"C","difficulty":"hard"}\n',
        encoding="utf-8",
    )

    merged_path = tmp_path / "merged.jsonl"
    result = runner.invoke(app, ["merge", str(chunk1), str(chunk2), "--output", str(merged_path)])

    assert result.exit_code == 0
    lines = merged_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "duplicates" in result.output.lower()
