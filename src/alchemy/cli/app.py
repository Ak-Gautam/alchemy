"""Alchemy CLI — generate synthetic datasets from natural language descriptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.syntax import Syntax

from alchemy.config.loader import load_pipeline_config
from alchemy.config.settings import ModelConfig, PipelineConfig
from alchemy.pipeline.context import PipelineContext
from alchemy.pipeline.plan import GenerationPlan
from alchemy.pipeline.engine import PipelineEngine
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


def _build_recipe_prompt_suffix(recipe_name: str) -> str:
    recipe = get_recipe(recipe_name)
    plan = recipe.build_plan()
    variation_axes = [axis.name for axis in plan.variation_spec.axes]
    quality_lines = "\n".join(f"- {item}" for item in plan.quality_rubric) or "- (none)"
    return (
        "\n\nRecipe constraints:\n"
        f"- recipe_name: {recipe.name}\n"
        f"- recipe_description: {recipe.description}\n"
        f"- row_schema: {json.dumps(plan.row_schema, ensure_ascii=False)}\n"
        f"- variation_axes: {', '.join(variation_axes) if variation_axes else '(none)'}\n"
        f"- quality_rubric:\n{quality_lines}"
    )


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
    recipe: Optional[str] = typer.Option(
        None, "--recipe", help="Built-in recipe name (e.g. text_compression_pairs)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to pipeline YAML config"
    ),
    planner: Optional[str] = typer.Option(
        None, "--planner", help="Planner model (provider:model_id)"
    ),
    generator: Optional[str] = typer.Option(
        None, "--generator", help="Generator model (provider:model_id)"
    ),
    validator: Optional[str] = typer.Option(
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
    if recipe:
        final_prompt += _build_recipe_prompt_suffix(recipe)

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
    ctx = engine.run(final_prompt, num_samples=final_num_samples)

    console.print(f"\n[bold green]Done![/] {ctx.metrics.get('valid_count', 0)} valid samples.")
    if ctx.artifact_paths:
        console.print(f"[cyan]Artifacts:[/] {ctx.artifact_paths.get('artifacts_dir')}")
    hf_path = ctx.artifact_paths.get("hf_dataset_path")
    if hf_path:
        console.print(f"[cyan]HF postprocess:[/] {hf_path}")


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Description of the dataset"),
    recipe: Optional[str] = typer.Option(
        None, "--recipe", help="Built-in recipe name (e.g. text_compression_pairs)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to pipeline YAML config"
    ),
    planner: Optional[str] = typer.Option(
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

    final_prompt = prompt
    if recipe:
        final_prompt += _build_recipe_prompt_suffix(recipe)

    engine = PipelineEngine.from_config(config)
    generation_plan = engine.run_plan_only(final_prompt)

    console.print("\n[bold]Generation Plan:[/]\n")
    plan_json = generation_plan.model_dump_json(indent=2)
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
    config_file: Optional[Path] = typer.Option(
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
        with plan_file.open("r", encoding="utf-8") as file_obj:
            plan_data: dict[str, Any] = json.load(file_obj)
        ctx.plan = GenerationPlan.from_dict(plan_data)
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
