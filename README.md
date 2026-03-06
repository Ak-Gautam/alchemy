# Alchemy

Alchemy is a local-first synthetic data generation harness optimized for Apple Silicon (MLX).
It uses a multi-agent pipeline (`planner -> generator -> validator`) and produces reproducible run artifacts.

## Features

- Local-first generation with `mlx-lm` models.
- Per-agent model config with split provider init and generation settings.
- Structural validation + LLM validation + exact deduplication.
- Robust JSON parsing for noisy model outputs (fences, `<think>`, trailing text).
- Built-in recipe framework with `text_compression_pairs`.
- Run artifacts:
  - `accepted.jsonl`
  - `rejected.jsonl`
  - `metrics.json`
  - `plan.json`
  - `resolved_config.yaml`
  - `report.md`
- Optional JSONL -> Hugging Face dataset postprocess.

## Requirements

- Python `>=3.14`
- Apple Silicon + MLX runtime for local generation
- `uv` for environment and dependency management

## Install

```bash
uv sync
```

Run tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

## CLI

### Generate

```bash
uv run alchemy generate "Generate chemistry QA pairs" \
  --planner mlx:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --generator mlx:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --validator mlx:mlx-community/Qwen2.5-7B-Instruct-4bit \
  --format json \
  --output ./output/chemistry.jsonl
```

Recipe-aware generation:

```bash
uv run alchemy generate "Generate compression training data" \
  --recipe text_compression_pairs \
  --format json \
  --output ./output/compression.jsonl
```

Enable HF postprocess from artifact JSONL:

```bash
uv run alchemy generate "Generate compression training data" \
  --format json \
  --output ./output/compression.jsonl \
  --postprocess-hf-from-jsonl \
  --postprocess-hf-output ./output/compression_hf
```

### Plan

```bash
uv run alchemy plan "Generate scientific abstract datasets" --recipe text_compression_pairs
```

### Validate

Validate using a saved plan file:

```bash
uv run alchemy validate ./output/dataset.jsonl --plan-file ./output/dataset_artifacts/plan.json
```

Or regenerate plan from a prompt:

```bash
uv run alchemy validate ./output/dataset.jsonl --prompt "Generate chemistry QA pairs"
```

### Recipes

```bash
uv run alchemy recipes
```

## Configuration

Use a YAML file and pass it with `--config`.

```yaml
planner_model:
  provider_type: mlx
  model_id: mlx-community/Qwen2.5-7B-Instruct-4bit
  init:
    max_context_tokens: 8192
  generation:
    temperature: 0.2
    max_tokens: 2048

generator_model:
  provider_type: mlx
  model_id: mlx-community/Qwen2.5-7B-Instruct-4bit
  init: {}
  generation:
    temperature: 0.8
    top_p: 0.95
    max_tokens: 4096

validator_model:
  provider_type: mlx
  model_id: mlx-community/Qwen2.5-7B-Instruct-4bit
  init: {}
  generation:
    temperature: 0.0
    max_tokens: 1024

output_format: json
output_path: ./output/run.jsonl
num_samples: 500
batch_size: 10
min_quality_score: 0.7

postprocess_hf_from_jsonl: true
postprocess_hf_output_path: ./output/run_hf
```

Legacy `options` under model config is still supported and migrated to `init`.

## Development Notes

- Current built-in recipe set is intentionally small while core pipeline hardens.
- Schema-first planner path exists (`SchemaPlannerAgent` + `alchemy.spec`) and can be wired into the main engine incrementally.
