"""Alchemy CLI — generate synthetic datasets from natural language descriptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax

from alchemy.config.loader import load_pipeline_config
from alchemy.config.settings import ModelConfig, PipelineConfig
from alchemy.pipeline.artifacts import write_run_artifacts
from alchemy.pipeline.context import PipelineContext
from alchemy.pipeline.datasets import merge_rows, write_rows
from alchemy.pipeline.engine import PipelineEngine
from alchemy.pipeline.plans import PlanType, load_plan, set_plan_total_rows
from alchemy.recipes import get_recipe, list_recipe_names

app = typer.Typer(
    name="alchemy",
    help="Synthetic data generation agent harness.",
    add_completion=False,
)
console = Console()


def _parse_model_string(model_str: str) -> ModelConfig:
    """Parse 'provider:model_id' string into ModelConfig.

    Examples:
        'mlx:mlx-community/Qwen2.5-7B-Instruct-4bit'
        'openai:gpt-5.3'
        'google:gemini-3.1-pro'
    """
    if ":" in model_str:
        provider, model_id = model_str.split(":", 1)
    else:
        # Default to mlx for unqualified model IDs
        provider, model_id = "mlx", model_str
    return ModelConfig(provider_type=provider, model_id=model_id)


def _build_recipe_plan(
    recipe_name: str,
    *,
    user_prompt: str,
    num_samples: int,
    batch_size: int,
) -> PlanType:
    recipe = get_recipe(recipe_name)
    plan = recipe.build_plan(num_rows=num_samples, batch_size=batch_size)
    plan.description = f"{recipe.description} User intent: {user_prompt}"
    plan.metadata = {**plan.metadata, "user_prompt": user_prompt}
    return plan


def _load_plan_file(path: Path) -> PlanType:
    with path.open("r", encoding="utf-8") as file_obj:
        plan_data: dict[str, Any] = json.load(file_obj)
    return load_plan(plan_data)


def _resolve_run_sizes(
    config: PipelineConfig,
    recipe_name: str | None,
    num_samples: int | None,
    batch_size: int | None,
) -> tuple[int, int]:
    if recipe_name is None:
        return (
            num_samples if num_samples is not None else config.num_samples,
            batch_size if batch_size is not None else config.batch_size,
        )

    recipe_plan = get_recipe(recipe_name).build_plan()
    return (
        num_samples if num_samples is not None else recipe_plan.defaults.num_rows,
        batch_size if batch_size is not None else recipe_plan.defaults.batch_size,
    )


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Description of the dataset to generate"),
    num_samples: int | None = typer.Option(None, "--num-samples", "-n", help="Number of samples"),
    output_format: str = typer.Option(
        "huggingface", "--format", "-f", help="Output format: json, code, huggingface"
    ),
    output_path: str = typer.Option("./output", "--output", "-o", help="Output path"),
    recipe: str | None = typer.Option(
        None, "--recipe", help="Built-in recipe name (e.g. text_compression_pairs)"
    ),
    plan_file: Path | None = typer.Option(
        None,
        "--plan-file",
        help="Saved plan JSON to reuse for chunked generation or repeatable runs",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to pipeline YAML config"
    ),
    planner: str | None = typer.Option(
        None, "--planner", help="Planner model (provider:model_id)"
    ),
    generator: str | None = typer.Option(
        None, "--generator", help="Generator model (provider:model_id)"
    ),
    validator: str | None = typer.Option(
        None, "--validator", help="Validator model (provider:model_id)"
    ),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Samples per batch"),
    min_quality_score: float | None = typer.Option(
        None, "--min-quality-score", help="Validation acceptance threshold (0-1)"
    ),
    postprocess_hf_from_jsonl: bool = typer.Option(
        False,
        "--postprocess-hf-from-jsonl",
        help="After run artifacts are written, convert accepted.jsonl to HF dataset",
    ),
    postprocess_hf_output_path: str | None = typer.Option(
        None,
        "--postprocess-hf-output",
        help="Output path for HF postprocess dataset (requires --postprocess-hf-from-jsonl)",
    ),
    chunk_id: str | None = typer.Option(
        None,
        "--chunk-id",
        help="Optional chunk/session identifier to bias each run toward distinct rows",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Generate a synthetic dataset from a natural language description."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load base config
    config = load_pipeline_config(config_file) if config_file else PipelineConfig()
    final_num_samples, final_batch_size = _resolve_run_sizes(
        config=config,
        recipe_name=recipe,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    final_prompt = prompt
    plan_override: PlanType | None = None
    if recipe and plan_file is not None:
        console.print("[red]Error:[/] --recipe and --plan-file are mutually exclusive")
        raise typer.Exit(1)
    if recipe:
        plan_override = _build_recipe_plan(
            recipe,
            user_prompt=prompt,
            num_samples=final_num_samples,
            batch_size=final_batch_size,
        )
    elif plan_file is not None:
        plan_override = _load_plan_file(plan_file)
        set_plan_total_rows(plan_override, final_num_samples)

    # Apply CLI overrides
    config.output_format = output_format
    config.output_path = output_path
    config.num_samples = final_num_samples
    config.batch_size = final_batch_size
    if min_quality_score is not None:
        config.min_quality_score = min_quality_score
    config.postprocess_hf_from_jsonl = postprocess_hf_from_jsonl
    if postprocess_hf_output_path is not None:
        config.postprocess_hf_output_path = postprocess_hf_output_path
    if planner:
        config.planner_model = _parse_model_string(planner)
    if generator:
        config.generator_model = _parse_model_string(generator)
    if validator:
        config.validator_model = _parse_model_string(validator)

    engine = PipelineEngine.from_config(config)
    ctx = engine.run(
        final_prompt,
        num_samples=final_num_samples,
        plan=plan_override,
        chunk_id=chunk_id,
    )

    console.print(f"\n[bold green]Done![/] {ctx.metrics.get('valid_count', 0)} valid samples.")
    if ctx.artifact_paths:
        console.print(f"[cyan]Artifacts:[/] {ctx.artifact_paths.get('artifacts_dir')}")
    hf_path = ctx.artifact_paths.get("hf_dataset_path")
    if hf_path:
        console.print(f"[cyan]HF postprocess:[/] {hf_path}")


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Description of the dataset"),
    recipe: str | None = typer.Option(
        None, "--recipe", help="Built-in recipe name (e.g. text_compression_pairs)"
    ),
    num_samples: int | None = typer.Option(None, "--num-samples", "-n", help="Planned row count"),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Planned batch size"),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional file path to save the plan JSON",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to pipeline YAML config"
    ),
    planner: str | None = typer.Option(
        None, "--planner", help="Planner model (provider:model_id)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run only the planning phase — inspect the schema without generating data."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = load_pipeline_config(config_file) if config_file else PipelineConfig()
    if planner:
        config.planner_model = _parse_model_string(planner)

    engine = PipelineEngine.from_config(config)
    plan_override: PlanType | None = None
    if recipe:
        resolved_num_samples, resolved_batch_size = _resolve_run_sizes(
            config=config,
            recipe_name=recipe,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        plan_override = _build_recipe_plan(
            recipe,
            user_prompt=prompt,
            num_samples=resolved_num_samples,
            batch_size=resolved_batch_size,
        )
    generation_plan = engine.run_plan_only(prompt, plan=plan_override)

    plan_json = generation_plan.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(plan_json, encoding="utf-8")
        console.print(f"[bold green]Saved plan:[/] {output_path}")
        return

    console.print("\n[bold]Generation Plan:[/]\n")
    console.print(Syntax(plan_json, "json", theme="monokai"))


@app.command()
def validate(
    input_path: Path = typer.Argument(..., help="Path to dataset to validate"),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Prompt to regenerate plan when --plan-file is not provided"
    ),
    plan_file: Path | None = typer.Option(
        None, "--plan-file", help="Path to saved plan JSON (from artifacts/plan.json)"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to pipeline YAML config with schema"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Validate an existing dataset against a plan."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not input_path.exists():
        console.print(f"[red]Error:[/] {input_path} does not exist")
        raise typer.Exit(1)

    # Load samples from JSONL
    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    console.print(f"Loaded {len(samples)} samples from {input_path}")

    if plan_file is None and prompt is None:
        console.print("[red]Error:[/] provide either --plan-file or --prompt for validation")
        raise typer.Exit(1)

    config = load_pipeline_config(config_file) if config_file else PipelineConfig()
    engine = PipelineEngine.from_config(config)

    ctx = PipelineContext(user_prompt="validation")
    if plan_file is not None:
        ctx.plan = _load_plan_file(plan_file)
    else:
        assert prompt is not None
        ctx.plan = engine.run_plan_only(prompt)
    ctx.raw_samples = samples
    console.print(f"Validating {len(samples)} samples...")
    ctx.validated_samples, ctx.rejected_samples = engine._run_validation(ctx)
    engine._run_deduplication(ctx)
    console.print(
        f"[bold green]Validation complete.[/] "
        f"{len(ctx.validated_samples)} valid, {len(ctx.rejected_samples)} rejected."
    )


@app.command()
def merge(
    inputs: list[Path] = typer.Argument(..., help="Chunk outputs to merge"),
    output_path: str = typer.Option(
        "./output/merged.jsonl",
        "--output",
        "-o",
        help="Merged output path",
    ),
    output_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Merged output format: json, jsonl, huggingface, hf",
    ),
    plan_file: Path | None = typer.Option(
        None,
        "--plan-file",
        help="Optional plan JSON used to structurally validate merged rows",
    ),
) -> None:
    """Merge chunked datasets into one deduplicated dataset."""
    plan = _load_plan_file(plan_file) if plan_file is not None else None
    merged_rows, duplicate_rows, invalid_rows = merge_rows(inputs, plan=plan)
    final_output_path = write_rows(
        rows=merged_rows,
        output_path=output_path,
        output_format=output_format,
        plan=plan,
    )

    metrics = {
        "raw_sample_count": len(merged_rows) + len(duplicate_rows) + len(invalid_rows),
        "valid_count": len(merged_rows),
        "rejected_count": len(duplicate_rows) + len(invalid_rows),
        "duplicate_count": len(duplicate_rows),
        "invalid_count": len(invalid_rows),
    }
    artifact_paths = write_run_artifacts(
        output_path=final_output_path,
        accepted_samples=merged_rows,
        rejected_samples=duplicate_rows + invalid_rows,
        metrics=metrics,
        plan=plan.model_dump() if plan is not None else None,
    )

    console.print(f"[bold green]Merged.[/] {len(merged_rows)} rows written to {final_output_path}")
    console.print(
        f"[cyan]Rejected:[/] {len(duplicate_rows)} duplicates, {len(invalid_rows)} invalid"
    )
    console.print(f"[cyan]Artifacts:[/] {artifact_paths['artifacts_dir']}")


@app.command("recipes")
def list_recipes() -> None:
    """List available built-in recipes."""
    names = list_recipe_names()
    if not names:
        console.print("No built-in recipes registered.")
        return
    console.print("[bold]Built-in recipes:[/]")
    for name in names:
        console.print(f"- {name}")


def main() -> None:
    app()
