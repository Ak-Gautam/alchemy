# Alchemy

Alchemy is a local-first synthetic data generation harness optimized for Apple Silicon.
It uses a schema-first multi-agent pipeline (`planner -> generator -> validator`) and
produces reproducible run artifacts.

## Features

- Local-first generation with either `llama.cpp` GGUF models or `mlx-lm` models.
- Per-agent model config with split provider init and generation settings.
- JSON Schema structural validation + LLM validation + exact deduplication.
- Robust JSON parsing for noisy model outputs (fences, `<think>`, trailing text).
- Built-in recipe framework with `text_compression_pairs`.
- Chunk-friendly generation via saved plans and `--chunk-id`.
- Merge command for deduplicating many chunk outputs into one dataset.
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
- Apple Silicon
- Either:
  - local `llama.cpp` server for GGUF inference, or
  - MLX runtime for `mlx-lm` models
- `uv` for environment and dependency management

## Install

```bash
uv sync
```

## Preferred Apple Silicon path

For Unsloth GGUF models on Apple Silicon, prefer `llama.cpp` + Metal over model conversion to MLX. The server is OpenAI-compatible, so Alchemy can use it through the existing chat provider path.

Example server launch:

```bash
llama-server \
  --model "$HOME/model_storage/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
  --host 127.0.0.1 \
  --port 8091 \
  --ctx-size 8192 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --jinja
```

Then point Alchemy at the local server with `provider_type: llamacpp` and `init.base_url: http://127.0.0.1:8091/v1`.

Run tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

## CLI

### Generate

```bash
uv run alchemy generate "Generate chemistry QA pairs" \
  --planner llamacpp:Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --generator llamacpp:Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --validator llamacpp:Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --config ./configs/examples/apple_silicon_gguf.yaml \
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

Repeatable chunked generation from a saved plan:

```bash
uv run alchemy plan "Generate chemistry QA pairs" --output ./output/chemistry_plan.json

uv run alchemy generate "Generate chemistry QA pairs" \
  --plan-file ./output/chemistry_plan.json \
  --num-samples 1000 \
  --chunk-id chunk-01 \
  --format json \
  --output ./output/chunks/chemistry_chunk_01.jsonl
```

Run more chunks with the same `--plan-file` and different `--chunk-id` values, then merge:

```bash
uv run alchemy merge \
  ./output/chunks/chemistry_chunk_01.jsonl \
  ./output/chunks/chemistry_chunk_02.jsonl \
  ./output/chunks/chemistry_chunk_03.jsonl \
  --plan-file ./output/chemistry_plan.json \
  --format json \
  --output ./output/chemistry_merged.jsonl
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
  provider_type: llamacpp
  model_id: Qwen3.5-9B-UD-Q4_K_XL.gguf
  init:
    base_url: http://127.0.0.1:8091/v1
    api_key: local
  generation:
    temperature: 0.2
    max_tokens: 2048

generator_model:
  provider_type: llamacpp
  model_id: Qwen3.5-9B-UD-Q4_K_XL.gguf
  init:
    base_url: http://127.0.0.1:8091/v1
    api_key: local
  generation:
    temperature: 0.8
    top_p: 0.95
    max_tokens: 4096

validator_model:
  provider_type: llamacpp
  model_id: Qwen3.5-9B-UD-Q4_K_XL.gguf
  init:
    base_url: http://127.0.0.1:8091/v1
    api_key: local
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

`provider_type: openai` still works for remote OpenAI-compatible endpoints. Use `provider_type: llamacpp` or `llama_cpp` when the backend is a local `llama-server`.

## Development Notes

- Current built-in recipe set is intentionally small while core pipeline hardens.
- Validation now fails closed: validator batch errors reject rows instead of accepting them.
