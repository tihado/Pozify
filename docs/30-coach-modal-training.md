# Coach Summary Modal Training

This guide explains the current Modal workflow for training, evaluating, merging, and publishing the
Pozify grounded coach-summary model.

## Scope

The pipeline lives in:

- `scripts/coach_summary_modal.py`

Available stages:

- `prepare-data`
- `train`
- `evaluate`
- `merge`
- `publish`
- `publish-merged`
- `all`

## What This Pipeline Produces

The pipeline fine-tunes a coach-summary model for Pozify's structured JSON-to-summary task.

It can produce:

- a LoRA adapter
- evaluation artifacts
- a merged full Transformers checkpoint
- a Hugging Face repo containing the merged checkpoint

Important limitation:

- Publishing a merged model repo to Hugging Face does **not** currently guarantee that the repo can
  be used through Hugging Face serverless `chat_completion`.
- In the current codebase, the safest cloud runtime remains `Qwen/Qwen3-14B`.
- The simplest way to use the fine-tuned merged model today is local inference through
  `POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR`.

## Requirements

1. Install and authenticate Modal:

```bash
uv run modal setup
```

2. Make sure local `.env` contains at least:

- `HF_TOKEN`
- optional `POZIFY_COACH_SUMMARY_HF_REPO_ID`
- optional `POZIFY_COACH_SUMMARY_MERGED_HF_REPO_ID`
- optional `POZIFY_COACH_SUMMARY_MODEL`
- optional `POZIFY_COACH_SUMMARY_HF_PRIVATE=1`

The script reads local `.env` and injects those values into a Modal secret at runtime.

## Input Data

The pipeline expects these files:

- `data/sft/coach_summary_train.jsonl`
- `data/sft/coach_summary_eval.jsonl`
- `data/sft/public_fitness_style.jsonl`
- `configs/coach_summary_lora.default.json`

The default checked-in config currently points to:

- base model: `Qwen/Qwen3-14B`

## Recommended Training Flow

### 1. Build the SFT dataset

```bash
uv run python scripts/build_coach_summary_sft_dataset.py
```

### 2. Copy data into the Modal data volume

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
```

### 3. Train the adapter

```bash
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 2 --style-weight 0.2
```

### 4. Evaluate

```bash
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
```

### 5. Merge the adapter into a full model

```bash
uv run modal run scripts/coach_summary_modal.py --stage merge
```

### 6. Publish the merged model

```bash
uv run modal run scripts/coach_summary_modal.py --stage publish-merged --repo-id build-small-hackathon/pozify-coach-summary1
```

### Full end-to-end run

```bash
uv run modal run scripts/coach_summary_modal.py --stage all --epochs 2 --style-weight 0.2 --repo-id build-small-hackathon/pozify-coach-summary1
```

## Cheap Smoke-Test Workflow

If you want to reduce GPU cost before a full run:

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 1 --style-weight 0.2
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
```

If the format and verifier behavior look acceptable, rerun with `--epochs 2` and then `merge` +
`publish-merged`.

## Model Volume Artifacts

The Modal model volume stores:

- `adapter/`
- `merged_model/`
- `training_config.json`
- `training_summary.json`
- `evaluation.json`
- `hf_upload.json`
- `merge_summary.json`
- `hf_merged_upload.json`
- `README.md`

## Current Runtime Usage

### Safest cloud runtime now

```bash
export POZIFY_COACH_SUMMARY_MODEL=Qwen/Qwen3-14B
uv run python app.py
```

### Use the merged fine-tuned model locally

Download the merged repo or copy the merged directory locally, then:

```bash
export POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR=/path/to/merged_model
export POZIFY_COACH_SUMMARY_BASE_MODEL=Qwen/Qwen3-14B
export POZIFY_COACH_SUMMARY_ADAPTER_ID=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

### Not recommended right now

Pointing app runtime directly at the merged Hugging Face repo through:

```bash
export POZIFY_COACH_SUMMARY_MODEL=build-small-hackathon/pozify-coach-summary1
```

This may still fall back because HF serverless currently rejects that repo as “not a chat model”.

## Evaluation Meaning

The coach-summary evaluation is task-specific. It measures:

- JSON validity rate
- verifier pass rate
- section completeness rate
- failure count and example failures

This is more useful for Pozify than generic text-generation metrics because the product depends on:

- schema correctness
- grounded issue references
- safe variation handling
- no diagnosis / no injury-prevention claims

## Related Docs

- [../README.md](../README.md)
- [31-coach-training-report.md](31-coach-training-report.md)
- [92-archive-coach-model-data-plan.md](92-archive-coach-model-data-plan.md)
