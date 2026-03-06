"""Pipeline engine — orchestrates planning, generation, validation, and output."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from alchemy.agents.generator import GeneratorAgent
from alchemy.agents.planner import PlannerAgent
from alchemy.agents.validator import ValidatorAgent
from alchemy.config.settings import PipelineConfig
from alchemy.models.base import GenerationConfig, ModelProvider
from alchemy.models.registry import create_provider
from alchemy.outputs.base import OutputAdapter
from alchemy.outputs.hf_postprocess import convert_jsonl_to_hf_dataset
from alchemy.quality.dedupe import dedupe_exact_rows
from alchemy.schemas.dynamic import validate_sample_structure

from .artifacts import write_run_artifacts
from .context import PipelineContext
from .plan import GenerationPlan

logger = logging.getLogger(__name__)
console = Console()


class PipelineEngine:
    """Main orchestrator for the plan -> generate -> validate -> output pipeline."""

    def __init__(
        self,
        planner_provider: ModelProvider,
        generator_provider: ModelProvider,
        validator_provider: ModelProvider,
        output_adapter: OutputAdapter,
        config: PipelineConfig | None = None,
        planner_generation_config: GenerationConfig | None = None,
        generator_generation_config: GenerationConfig | None = None,
        validator_generation_config: GenerationConfig | None = None,
    ):
        self.planner = PlannerAgent(
            planner_provider,
            generation_config=planner_generation_config,
        )
        self.generator = GeneratorAgent(
            generator_provider,
            generation_config=generator_generation_config,
        )
        self.validator = ValidatorAgent(
            validator_provider,
            generation_config=validator_generation_config,
        )
        self.output_adapter = output_adapter
        self.config = config or PipelineConfig()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> PipelineEngine:
        """Build the engine from a PipelineConfig."""
        planner_provider = create_provider(
            config.planner_model.provider_type,
            config.planner_model.model_id,
            **config.planner_model.init,
        )
        generator_provider = create_provider(
            config.generator_model.provider_type,
            config.generator_model.model_id,
            **config.generator_model.init,
        )
        validator_provider = create_provider(
            config.validator_model.provider_type,
            config.validator_model.model_id,
            **config.validator_model.init,
        )
        output_adapter = OutputAdapter.create(config.output_format, config.output_path)
        return cls(
            planner_provider,
            generator_provider,
            validator_provider,
            output_adapter,
            config,
            planner_generation_config=config.planner_model.generation,
            generator_generation_config=config.generator_model.generation,
            validator_generation_config=config.validator_model.generation,
        )

    def run(self, user_prompt: str, num_samples: int | None = None) -> PipelineContext:
        """Execute the full pipeline end-to-end."""
        ctx = PipelineContext(user_prompt=user_prompt)

        # Phase 1: Planning
        ctx.plan = self._run_planning(ctx)
        if num_samples is not None:
            ctx.plan.num_samples = num_samples

        # Phase 2: Generation
        ctx.raw_samples = self._run_generation(ctx)

        # Phase 3: Validation
        ctx.validated_samples, ctx.rejected_samples = self._run_validation(ctx)

        # Phase 4: Exact dedupe
        self._run_deduplication(ctx)

        # Phase 5: Output
        ctx.output_path = self._run_output(ctx)

        # Phase 6: Run artifacts
        ctx.artifact_paths = self._run_artifacts(ctx)

        self._print_summary(ctx)
        return ctx

    def run_plan_only(self, user_prompt: str) -> GenerationPlan:
        """Run only the planning phase and return the plan."""
        ctx = PipelineContext(user_prompt=user_prompt)
        return self._run_planning(ctx)

    def _run_planning(self, ctx: PipelineContext) -> GenerationPlan:
        console.print("\n[bold blue]Phase 1:[/] Planning dataset schema...")
        start = perf_counter()
        plan = self.planner.plan(ctx.user_prompt)
        elapsed = perf_counter() - start
        ctx.metrics["planning_seconds"] = round(elapsed, 2)
        console.print(
            f"  Dataset: [cyan]{plan.dataset_name}[/] — "
            f"{len(plan.fields)} fields, {plan.num_samples} samples planned"
        )
        return plan

    def _run_generation(self, ctx: PipelineContext) -> list[dict[str, Any]]:
        plan = ctx.plan
        assert plan is not None
        all_samples: list[dict[str, Any]] = []
        batch_size = plan.batch_size
        total_batches = (plan.num_samples + batch_size - 1) // batch_size

        console.print(
            f"\n[bold green]Phase 2:[/] Generating {plan.num_samples} samples "
            f"in {total_batches} batches..."
        )
        start = perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating", total=total_batches)
            for i in range(total_batches):
                remaining = plan.num_samples - len(all_samples)
                current_batch_size = min(batch_size, remaining)
                try:
                    batch = self.generator.generate_batch(
                        plan, batch_size=current_batch_size, batch_index=i
                    )
                    all_samples.extend(batch)
                except Exception as e:
                    logger.warning("Batch %d failed: %s", i, e)
                progress.advance(task)

        elapsed = perf_counter() - start
        ctx.metrics["generation_seconds"] = round(elapsed, 2)
        ctx.metrics["raw_sample_count"] = len(all_samples)
        console.print(f"  Generated {len(all_samples)} raw samples")
        return all_samples

    def _run_validation(
        self, ctx: PipelineContext
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        plan = ctx.plan
        assert plan is not None
        samples = ctx.raw_samples
        console.print(f"\n[bold yellow]Phase 3:[/] Validating {len(samples)} samples...")
        start = perf_counter()

        valid: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        # First pass: fast structural validation
        structurally_valid = []
        for sample in samples:
            issues = validate_sample_structure(sample, plan)
            if issues:
                rejected.append({"sample": sample, "issues": issues, "score": 0.0})
            else:
                structurally_valid.append(sample)

        if rejected:
            console.print(
                f"  Structural check: {len(rejected)} samples failed, "
                f"{len(structurally_valid)} passed"
            )

        # Second pass: LLM validation on structurally valid samples
        batch_size = plan.batch_size
        for i in range(0, len(structurally_valid), batch_size):
            batch = structurally_valid[i : i + batch_size]
            try:
                results = self.validator.validate_batch(plan, batch)
                for vr in results:
                    if vr.is_valid and vr.score >= self.config.min_quality_score:
                        valid.append(vr.sample)
                    else:
                        rejected.append(
                            {"sample": vr.sample, "issues": vr.issues, "score": vr.score}
                        )
            except Exception as e:
                logger.warning("Validation batch %d failed, accepting samples: %s", i, e)
                valid.extend(batch)

        elapsed = perf_counter() - start
        ctx.metrics["validation_seconds"] = round(elapsed, 2)
        ctx.metrics["valid_count"] = len(valid)
        ctx.metrics["rejected_count"] = len(rejected)
        console.print(f"  Passed: {len(valid)}, Rejected: {len(rejected)}")
        return valid, rejected

    def _run_output(self, ctx: PipelineContext) -> str:
        plan = ctx.plan
        assert plan is not None
        console.print(f"\n[bold magenta]Phase 5:[/] Writing output ({self.config.output_format})...")
        path = self.output_adapter.write(ctx.validated_samples, plan)
        console.print(f"  Saved to: {path}")
        return path

    def _print_summary(self, ctx: PipelineContext) -> None:
        console.print("\n[bold]Pipeline complete.[/]")
        m = ctx.metrics
        console.print(f"  Planning:    {m.get('planning_seconds', '?')}s")
        console.print(f"  Generation:  {m.get('generation_seconds', '?')}s")
        console.print(f"  Validation:  {m.get('validation_seconds', '?')}s")
        console.print(
            f"  Samples:     {m.get('valid_count', 0)} valid / "
            f"{m.get('raw_sample_count', 0)} generated"
        )
        if "duplicate_count" in m:
            console.print(f"  Duplicates:  {m.get('duplicate_count', 0)} removed")
        console.print(f"  Output:      {ctx.output_path}")
        artifacts_dir = ctx.artifact_paths.get("artifacts_dir")
        if artifacts_dir:
            console.print(f"  Artifacts:   {artifacts_dir}")

    def _run_artifacts(self, ctx: PipelineContext) -> dict[str, str]:
        output_path = ctx.output_path
        assert output_path is not None
        console.print("\n[bold cyan]Phase 6:[/] Writing run artifacts...")
        artifact_paths = write_run_artifacts(
            output_path=output_path,
            accepted_samples=ctx.validated_samples,
            rejected_samples=ctx.rejected_samples,
            metrics=ctx.metrics,
            plan=ctx.plan.model_dump() if ctx.plan is not None else None,
            resolved_config=self.config.model_dump(),
        )
        if self.config.postprocess_hf_from_jsonl:
            hf_path = self.config.postprocess_hf_output_path
            if not hf_path:
                hf_path = f"{artifact_paths['artifacts_dir']}/hf_dataset"
            artifact_paths["hf_dataset_path"] = convert_jsonl_to_hf_dataset(
                artifact_paths["accepted_jsonl"],
                hf_path,
            )
            console.print(f"  Saved HF postprocess dataset to: {artifact_paths['hf_dataset_path']}")
        console.print(f"  Saved run artifacts to: {artifact_paths['artifacts_dir']}")
        return artifact_paths

    def _run_deduplication(self, ctx: PipelineContext) -> None:
        console.print(
            f"\n[bold white]Phase 4:[/] Deduplicating {len(ctx.validated_samples)} validated samples..."
        )
        unique_rows, duplicate_rows = dedupe_exact_rows(ctx.validated_samples)
        duplicate_count = len(duplicate_rows)
        for sample in duplicate_rows:
            ctx.rejected_samples.append(
                {"sample": sample, "issues": ["duplicate_exact"], "score": 0.0}
            )
        ctx.validated_samples = unique_rows
        ctx.metrics["duplicate_count"] = duplicate_count
        ctx.metrics["valid_count"] = len(ctx.validated_samples)
        ctx.metrics["rejected_count"] = len(ctx.rejected_samples)
        console.print(
            f"  Unique after dedupe: {len(ctx.validated_samples)}, "
            f"duplicates removed: {duplicate_count}"
        )
