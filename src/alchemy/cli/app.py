"""Alchemy CLI — generate synthetic datasets from natural language descriptions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax

from alchemy.config.loader import load_pipeline_config
from alchemy.config.settings import ModelConfig, PipelineConfig
from alchemy.pipeline.engine import PipelineEngine

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


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Description of the dataset to generate"),
    num_samples: int = typer.Option(100, "--num-samples", "-n", help="Number of samples"),
    output_format: str = typer.Option(
        "huggingface", "--format", "-f", help="Output format: json, code, huggingface"
    ),
    output_path: str = typer.Option("./output", "--output", "-o", help="Output path"),
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
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Samples per batch"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Generate a synthetic dataset from a natural language description."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load base config
    config = load_pipeline_config(config_file) if config_file else PipelineConfig()

    # Apply CLI overrides
    config.output_format = output_format
    config.output_path = output_path
    config.num_samples = num_samples
    config.batch_size = batch_size
    if planner:
        config.planner_model = _parse_model_string(planner)
    if generator:
        config.generator_model = _parse_model_string(generator)
    if validator:
        config.validator_model = _parse_model_string(validator)

    engine = PipelineEngine.from_config(config)
    ctx = engine.run(prompt, num_samples=num_samples)

    console.print(f"\n[bold green]Done![/] {ctx.metrics.get('valid_count', 0)} valid samples.")


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Description of the dataset"),
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

    engine = PipelineEngine.from_config(config)
    generation_plan = engine.run_plan_only(prompt)

    console.print("\n[bold]Generation Plan:[/]\n")
    plan_json = generation_plan.model_dump_json(indent=2)
    console.print(Syntax(plan_json, "json", theme="monokai"))


@app.command()
def validate(
    input_path: Path = typer.Argument(..., help="Path to dataset to validate"),
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

    if not config_file:
        console.print("[red]Error:[/] --config is required for validation (must contain schema)")
        raise typer.Exit(1)

    config = load_pipeline_config(config_file)
    engine = PipelineEngine.from_config(config)

    # Re-run validation on loaded samples
    from alchemy.pipeline.context import PipelineContext

    ctx = PipelineContext(user_prompt="validation")
    ctx.plan = engine.run_plan_only("validation")
    ctx.raw_samples = samples
    # Directly call validation
    console.print(f"Validating {len(samples)} samples...")


def main() -> None:
    app()
