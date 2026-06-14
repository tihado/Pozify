# Coach Summary Modal Training

This guide explains how to train, evaluate, and publish the Pozify coach-summary LoRA adapter on [Modal](https://modal.com).

## What The Script Does

The Modal pipeline lives in:

- `scripts/coach_summary_modal.py`

It follows the same stage pattern as the exercise-router Modal pipeline:

- `prepare-data`
- `train`
- `evaluate`
- `publish`
- `all`

## Requirements

1. Install the Modal CLI locally.
2. Authenticate Modal:

```bash
modal setup
```

3. Make sure your local `.env` contains:

- `HF_TOKEN`
- optional `POZIFY_COACH_SUMMARY_HF_REPO_ID`
- optional `POZIFY_COACH_SUMMARY_HF_PRIVATE=1`

The script uses `modal.Secret.from_dotenv()`, so for local `modal run ...` usage you do not need to pre-create a hosted Modal secret just to run `prepare-data` or `train`.

If you prefer a hosted secret for team/shared environments, you can still create one manually in Modal and adapt the script later.

## Data Inputs

The script expects these local artifacts to already exist in the repo:

- `data/sft/coach_summary_train.jsonl`
- `data/sft/coach_summary_eval.jsonl`
- `data/sft/public_fitness_style.jsonl`
- `configs/coach_summary_lora.default.json`

These are copied into the Modal data volume during `prepare-data`.

## Stage Commands

Prepare training data:

```bash
modal run scripts/coach_summary_modal.py --stage prepare-data
```

Train the LoRA adapter:

```bash
modal run scripts/coach_summary_modal.py --stage train --epochs 2 --style-weight 0.2
```

Evaluate the trained adapter:

```bash
modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
```

Publish the adapter and metadata to Hugging Face:

```bash
modal run scripts/coach_summary_modal.py --stage publish --repo-id build-small-hackathon/pozify-coach-summary
```

Run the full pipeline end to end:

```bash
modal run scripts/coach_summary_modal.py --stage all --epochs 2 --style-weight 0.2 --repo-id build-small-hackathon/pozify-coach-summary
```

## Artifacts

The Modal model volume stores:

- `adapter/`
- `training_config.json`
- `training_summary.json`
- `evaluation.json`
- `hf_upload.json`
- `README.md`

## Runtime Usage

After publish, point Pozify runtime to the trained adapter by setting:

```bash
export POZIFY_COACH_SUMMARY_ADAPTER_ID=build-small-hackathon/pozify-coach-summary
```

If you download the adapter locally instead, use:

```bash
export POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR=/path/to/adapter
```
