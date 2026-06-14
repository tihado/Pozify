---
title: Pozify
emoji: "🏋️"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.17.3"
python_version: "3.10"
app_file: app.py
fullWidth: true
short_description: Small-model workout form review from short videos.
tags:
  - gradio
  - computer-vision
  - pose-estimation
  - fitness
  - video-analysis
  - llama-cpp
---

# Pozify

Pozify turns a short workout video into a structured form-review report:

- exercise detected
- reps counted
- variation and issue markers
- annotated video and clips
- grounded coach summary
- verifier-backed confidence and safety notes

The app is built as a small-model pipeline, not a giant end-to-end model. It combines transparent
pose analysis, a trained exercise router, exercise-specific rule logic, deterministic knowledge-card
retrieval, and a small summary model.

Pozify is not a medical device. It does not diagnose injuries, claim injury prevention, or replace a
qualified trainer, clinician, or physical therapist.

## Current Status

The current codebase supports:

- web app runtime through `app.py`
- trained exercise routing for `squat`, `push_up`, `shoulder_press`, and `unknown`
- grounded coach-summary generation from structured JSON artifacts
- verifier and conservative fallback summaries
- Modal training pipelines for both the exercise router and coach-summary model

The current safest cloud runtime for coach summary is:

- `Qwen/Qwen3-14B`

The current fine-tuned coach-summary Hugging Face repo can be trained, merged, and published, but
it is not yet cleanly usable through the current Hugging Face serverless `chat_completion` path.
If you want to use that fine-tuned model today, the simplest path is local merged-model inference
through `POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR`.

## Product Flow

```text
video + user profile
-> video QC
-> MediaPipe pose extraction
-> pose cleaning
-> exercise router
-> exercise-specific rep counter
-> per-rep analysis
-> variation detection
-> issue markers
-> annotated video renderer
-> grounded coach summary
-> verifier
-> final report
```

Supported router labels:

- `squat`
- `push_up`
- `shoulder_press`
- `unknown`

## Model Stack

| Component | Model or method | Trained here? | Runtime |
| --- | --- | --- | --- |
| Pose extraction | MediaPipe Pose Landmarker Lite | No | CPU / MediaPipe delegate |
| Exercise router | PyTorch BiLSTM over 30-frame pose windows | Yes | Torch |
| Router baseline | scikit-learn `HistGradientBoostingClassifier` | Yes | CPU fallback/reference |
| Rep counting | Exercise-specific state machines | No ML | CPU |
| Issue markers | Transparent rules over per-rep metrics | No ML | CPU |
| Coach summary | `Qwen/Qwen3-14B` by default | Base model only | HF Inference, local Transformers, or llama.cpp |
| Coach-summary fine-tune | LoRA / merged checkpoint pipeline on Modal | Yes | Local merged-model path recommended |
| Verifier | Deterministic safety and grounding checks | No ML | CPU |

The trained router is intentionally tiny:

| Artifact | Count |
| --- | ---: |
| BiLSTM router trainable params | 182,796 |
| Router input features per frame | 237 |
| Window length | 30 frames |
| Output classes | 4 |

## Run The App Locally

This repo uses a `src/` layout, but `uv` is configured with `package = false`, so the correct local
entrypoint is:

```bash
uv run python app.py
```

The app listens at:

```text
http://127.0.0.1:7860
```

### Mock vs Real Mode

By default:

- if no video is provided, Pozify uses mock mode
- if a real video is uploaded, Pozify runs the full analysis pipeline

Force mock mode:

```bash
POZIFY_MOCK_MODE=1 uv run python app.py
```

Force real mode:

```bash
POZIFY_MOCK_MODE=0 uv run python app.py
```

If you already have the MediaPipe task file locally:

```bash
POZIFY_MEDIAPIPE_POSE_MODEL=/path/to/pose_landmarker_lite.task \
POZIFY_MOCK_MODE=0 \
uv run python app.py
```

## Coach Summary Runtime Options

### 1. Simplest cloud path

Use the default supported cloud model:

```bash
export POZIFY_COACH_SUMMARY_MODEL=Qwen/Qwen3-14B
uv run python app.py
```

This is the easiest path if you want the UI to avoid immediate fallback caused by unsupported custom
HF serverless routing.

### 2. Use the fine-tuned merged model locally

Download the merged repo locally, then point Pozify at it:

```bash
export POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR=/absolute/path/to/merged_model
export POZIFY_COACH_SUMMARY_BASE_MODEL=Qwen/Qwen3-14B
export POZIFY_COACH_SUMMARY_ADAPTER_ID=build-small-hackathon/pozify-coach-summary1
uv run python app.py
```

This is the simplest way to use `build-small-hackathon/pozify-coach-summary1` today without adding a
dedicated inference endpoint.

### 3. llama.cpp

Pozify can send the coach-summary prompt to a local `llama-server` that exposes the
OpenAI-compatible `/v1/chat/completions` endpoint.

Example:

```bash
llama-server \
  --model /path/to/qwen3-14b-instruct.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Then:

```bash
POZIFY_COACH_SUMMARY_PROVIDER=llama_cpp \
POZIFY_COACH_SUMMARY_MODEL=local-qwen3-14b-gguf \
POZIFY_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
POZIFY_COACH_SUMMARY_MAX_TOKENS=700 \
uv run python app.py
```

### Useful environment variables

| Variable | Purpose |
| --- | --- |
| `POZIFY_ROUTER_DEVICE` | Override router device, for example `cpu` or `cuda`. |
| `POZIFY_SPACES_GPU_DURATION` | `spaces.GPU` duration in seconds, default `120`. |
| `POZIFY_COACH_SUMMARY_PROVIDER` | `hf_inference`, `local_transformers`, or `llama_cpp`. |
| `POZIFY_COACH_SUMMARY_MODEL` | Coach model id or llama.cpp model alias. |
| `POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR` | Prefer a local merged/model directory for coach summary. |
| `POZIFY_COACH_SUMMARY_BYPASS_VERIFIER` | Keep model output even when verifier fails. |

## Exercise Router Training

Run the full router training and publish flow:

```bash
uv run modal run scripts/exercise_router_modal.py \
  --stage all \
  --repo-id build-small-hackathon/pozify-exercise-router
```

Step-by-step:

```bash
uv run modal run scripts/exercise_router_modal.py --stage ingest
uv run modal run scripts/exercise_router_modal.py --stage features
uv run modal run scripts/exercise_router_modal.py --stage train-baseline
uv run modal run scripts/exercise_router_modal.py --stage train-temporal
uv run modal run scripts/exercise_router_modal.py --stage evaluate
uv run modal run scripts/exercise_router_modal.py --stage publish --repo-id build-small-hackathon/pozify-exercise-router
```

The active router artifact is `temporal.pt`; the baseline is retained for comparison and fallback.

## Coach Summary Training

Build the grounded SFT dataset:

```bash
uv run python scripts/build_coach_summary_sft_dataset.py
```

Run the full coach-summary Modal flow:

```bash
uv run modal run scripts/coach_summary_modal.py \
  --stage all \
  --epochs 2 \
  --style-weight 0.2 \
  --repo-id build-small-hackathon/pozify-coach-summary1
```

Step-by-step:

```bash
uv run modal run scripts/coach_summary_modal.py --stage prepare-data
uv run modal run scripts/coach_summary_modal.py --stage train --epochs 2 --style-weight 0.2
uv run modal run scripts/coach_summary_modal.py --stage evaluate --limit 5
uv run modal run scripts/coach_summary_modal.py --stage merge
uv run modal run scripts/coach_summary_modal.py --stage publish-merged --repo-id build-small-hackathon/pozify-coach-summary1
```

Important runtime note:

- publishing the merged repo does not automatically make it usable through HF serverless
  `chat_completion`
- if you want immediate working inference, use `Qwen/Qwen3-14B`
- if you want to use the fine-tuned model, use `POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR`

## Generated Artifacts

Each run creates `runs/<run_id>/` with:

- `manifest.json`
- `user_profile.json`
- `video_manifest.json`
- `pose_sequence.json`
- `exercise_classification.json`
- `reps.json`
- `rep_debug.json`
- `rep_analysis.json`
- `variation.json`
- `issue_markers.json`
- `annotated_video.mp4`
- `coach_summary.json`
- `verification.json`
- `final_report.json`

JSON artifacts are validated before they are written. The final report records:

- analysis mode
- pose source
- knowledge-card provenance
- coach summary provider/model/source
- verifier status and bypass flags

## Docs Map

See [docs/01-docs-index.md](docs/01-docs-index.md) for the ordered documentation map.

Most useful operational docs:

- [docs/10-overview-build-small-hackathon-report.md](docs/10-overview-build-small-hackathon-report.md)
- [docs/20-router-training-report.md](docs/20-router-training-report.md)
- [docs/21-router-huggingface-release.md](docs/21-router-huggingface-release.md)
- [docs/30-coach-modal-training.md](docs/30-coach-modal-training.md)
- [docs/31-coach-training-report.md](docs/31-coach-training-report.md)
- [docs/40-data-custom-collection-guide.md](docs/40-data-custom-collection-guide.md)

## Project Structure

```text
app.py
web/
src/pozify/
  pipeline.py
  contracts.py
  steps/
  ml/
  slm/
  exercises/
scripts/
docs/
demo/
runs/
```

## Development Checks

```bash
uv run ruff check
uv run python -m compileall src scripts tests app.py
uv run python -m unittest discover -s tests
```

Run the real MediaPipe fixture smoke test only when the fixture is available:

```bash
POZIFY_RUN_REAL_POSE_TESTS=1 \
uv run python -m unittest tests.test_pose_steps.PoseStepTests.test_real_sample_mov_extracts_pose_landmarks
```
