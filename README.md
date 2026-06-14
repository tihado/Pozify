---
title: Pozify
emoji: 🏋️
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

Pozify is a Gradio Space that turns a short workout clip into a structured form-review report:
exercise detected, reps counted, issues localized by timestamp, an annotated video, and grounded
coach feedback.

The project was built for the
[Hugging Face Build Small Hackathon](https://huggingface.co/build-small-hackathon). It uses small,
replaceable models instead of a giant end-to-end model: pose extraction, a tiny trained exercise
router, rule-based rep and issue analysis, and a small instruction model for the final explanation.

## Hackathon Fit

| Requirement or badge | Pozify status |
| -------------------- | ------------- |
| Gradio app on HF Space | Uses `gr.Server` with a custom static frontend and FastAPI routes in `app.py`. |
| Small models only | Main coach model is Qwen2.5-7B Instruct; trained router has 182,796 params. Total is far below the 32B cap. |
| Well-Tuned | The exercise router was trained on pose-window data and published as `build-small-hackathon/pozify-exercise-router`. |
| Modal-powered | Dataset ingestion, pose feature extraction, training, evaluation, and Hub publishing run on Modal. |
| Llama Champion | Coach summary can run through `llama-server` with `POZIFY_COACH_SUMMARY_PROVIDER=llama_cpp`. |
| Field Notes | See [docs/build-small-hackathon-report.md](docs/build-small-hackathon-report.md). |
| Off-Brand | The app uses a custom frontend under `web/`, not the default Gradio blocks layout. |

## What It Does

```text
user profile + input video
-> video quality check
-> MediaPipe pose landmarks
-> pose cleaning and normalization
-> trained exercise router
-> exercise-specific rep counter
-> per-rep metrics and variation detection
-> frame-level issue markers
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

Pozify is not a medical device. It does not diagnose injuries, claim injury prevention, or replace a
trainer, clinician, or physical therapist.

## Model Stack

| Component | Model or method | Trained here? | Runtime |
| --------- | --------------- | ------------- | ------- |
| Pose extraction | MediaPipe Pose Landmarker Lite | No | CPU, MediaPipe Tasks GPU delegate when available |
| Exercise router | PyTorch BiLSTM over 30-frame pose windows | Yes | Torch CPU/CUDA, HF ZeroGPU wrapped |
| Router baseline | scikit-learn `HistGradientBoostingClassifier` | Yes | CPU fallback/reference |
| Rep counting | Exercise-specific state machines over pose signals | No ML | CPU |
| Issue markers | Transparent rules over per-rep metrics | No ML | CPU |
| Coach summary | `Qwen/Qwen2.5-7B-Instruct` by default | No | HF Inference, local Transformers, or llama.cpp |
| Verifier | Deterministic checks over generated JSON | No ML | CPU |

The trained router is intentionally tiny:

| Artifact | Count |
| -------- | ----: |
| BiLSTM router trainable params | 182,796 |
| Router input features per frame | 237 |
| Window length | 30 frames |
| Output classes | 4 |

## Training Data

The router starts from
[`RickyRiccio/Real_Time_Exercise_Recognition_Dataset`](https://huggingface.co/datasets/RickyRiccio/Real_Time_Exercise_Recognition_Dataset).
Supported folders are normalized to `squat`, `push_up`, and `shoulder_press`. Unsupported classes,
such as curl variations, are mapped to `unknown` so the router learns when to reject clips rather
than force a supported label.

Latest Modal training run:

| Metric | Value |
| ------ | ----: |
| Feature examples | 134 |
| Pose windows | 2,224 |
| Failed feature extractions | 0 |
| Squat windows | 659 |
| Push-up windows | 287 |
| Shoulder press windows | 646 |
| Unknown windows | 632 |

Feature extraction converts each video to cleaned pose windows:

- normalized COCO-17 landmarks with visibility
- knee, hip, elbow, and shoulder angles
- relative distances such as hand width over shoulder width
- frame deltas and velocities
- 30-frame tensors for the BiLSTM
- aggregated vectors for the scikit-learn baseline

Custom negative examples can be uploaded to the Modal data volume at `/data/raw/custom_unknown/`.
The collection protocol is documented in
[docs/custom-data-collection-guide.md](docs/custom-data-collection-guide.md).

## Training On Modal

Modal is used because the expensive work is bursty: download videos, run pose extraction over a
dataset, train/evaluate, publish artifacts, then shut down.

```bash
uv run modal setup
uv run modal run scripts/exercise_router_modal.py --stage ingest
uv run modal run scripts/exercise_router_modal.py --stage features
uv run modal run scripts/exercise_router_modal.py --stage train-baseline
uv run modal run scripts/exercise_router_modal.py --stage train-temporal
uv run modal run scripts/exercise_router_modal.py --stage evaluate
```

Or run the full flow:

```bash
uv run modal run scripts/exercise_router_modal.py \
  --stage all \
  --repo-id build-small-hackathon/pozify-exercise-router
```

Modal resources:

- `pozify-router-data`: raw dataset, manifests, extracted pose-window feature caches.
- `pozify-router-models`: `baseline.joblib`, `temporal.pt`, metrics, selection JSON, HF upload log.
- `train_temporal`: runs on Modal `A10` GPU.

Temporal router hyperparameters:

| Hyperparameter | Value |
| -------------- | ----: |
| Architecture | BiLSTM |
| Epochs | 73 |
| Hidden units | 73 |
| Dropout | 0.2174 |
| Learning rate | 0.0004 |
| Batch size | 54 |
| Final training loss | 0.0003 |

Current evaluation summary:

| Model | Artifact | Accuracy | Unknown rejection rate |
| ----- | -------- | -------: | ---------------------: |
| Baseline | `baseline.joblib` | 0.9982 | 0.9968 |
| BiLSTM temporal | `temporal.pt` | 0.9969 | 0.9968 |

The active runtime artifact is `temporal.pt`; the baseline is retained for comparison and fallback.
Full metrics are in
[docs/exercise-router-training-report.md](docs/exercise-router-training-report.md).

## Run The App Locally

Install dependencies with `uv`, then start the app:

```bash
uv run python app.py
```

The app listens at:

```text
http://127.0.0.1:7860
```

You can also run:

```bash
uv run gradio app.py
```

Real video mode is the default when a video path is provided. No-video demo runs use mock pose data
so the app can still produce deterministic JSON artifacts.

Force mock mode:

```bash
POZIFY_MOCK_MODE=1 uv run python app.py
```

Force real mode:

```bash
POZIFY_MOCK_MODE=0 uv run python app.py
```

MediaPipe downloads `pose_landmarker_lite.task` to `/tmp/pozify-models` on first use. To provide a
pre-downloaded task file:

```bash
POZIFY_MEDIAPIPE_POSE_MODEL=/path/to/pose_landmarker_lite.task \
POZIFY_MOCK_MODE=0 \
uv run python app.py
```

## Router Artifacts From Hugging Face

Pozify loads the router from this model repo by default:

```text
build-small-hackathon/pozify-exercise-router
```

Override the repo:

```bash
export POZIFY_ROUTER_HF_REPO_ID=owner/other-pozify-router
```

Disable Hub loading and use local files only:

```bash
export POZIFY_ROUTER_DISABLE_HF=1
```

Local active artifacts should live under:

```text
models/exercise_router/active/
  router_selection.json
  temporal.pt
  baseline.joblib        # optional fallback/reference
  router.joblib          # optional baseline alias
```

Publishing details are in
[docs/huggingface-router-release.md](docs/huggingface-router-release.md), and the model card draft is
in [docs/huggingface-router-model-card.md](docs/huggingface-router-model-card.md).

## GPU Runtime

### Hugging Face ZeroGPU

On a ZeroGPU Space, compute-heavy calls are wrapped with `spaces.GPU`. Hugging Face sets
`SPACES_ZERO_GPU=1`; Pozify then defaults the Torch router to CUDA.

Useful environment variables:

| Variable | Purpose |
| -------- | ------- |
| `POZIFY_ROUTER_DEVICE` | Override router device, for example `cpu` or `cuda`. |
| `POZIFY_SPACES_GPU_DURATION` | `spaces.GPU` duration in seconds, default `120`. |
| `POZIFY_COACH_SUMMARY_PROVIDER` | `hf_inference`, `local_transformers`, or `llama_cpp`. |
| `POZIFY_COACH_SUMMARY_MODEL` | Coach model id or llama.cpp model alias. |

Fully local Transformers coach mode on ZeroGPU:

```bash
POZIFY_COACH_SUMMARY_PROVIDER=local_transformers \
POZIFY_COACH_SUMMARY_MODEL=Qwen/Qwen2.5-7B-Instruct \
POZIFY_SPACES_GPU_DURATION=300 \
uv run python app.py
```

### Local CUDA

If Torch sees a CUDA GPU:

```bash
POZIFY_ROUTER_DEVICE=cuda \
POZIFY_COACH_SUMMARY_PROVIDER=local_transformers \
POZIFY_COACH_SUMMARY_MODEL=Qwen/Qwen2.5-7B-Instruct \
uv run python app.py
```

The router is small enough that CPU inference is usually fine; GPU is more useful for local
Transformers coach generation.

### llama.cpp

Pozify can send the coach-summary prompt to a local `llama-server` that exposes the
OpenAI-compatible `/v1/chat/completions` endpoint. This is the path for running the coach model as a
GGUF with llama.cpp.

Start a llama.cpp server with GPU offload. Use either a local GGUF file:

```bash
llama-server \
  --model /path/to/qwen2.5-7b-instruct-q4_k_m.gguf \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Or let llama.cpp download a GGUF repo from Hugging Face:

```bash
llama-server \
  --hf-repo owner/qwen2.5-7b-instruct-gguf:Q4_K_M \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --host 127.0.0.1 \
  --port 8080
```

Then run Pozify against that server:

```bash
POZIFY_COACH_SUMMARY_PROVIDER=llama_cpp \
POZIFY_COACH_SUMMARY_MODEL=local-qwen2.5-7b-gguf \
POZIFY_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
POZIFY_COACH_SUMMARY_MAX_TOKENS=700 \
uv run python app.py
```

Use `--n-gpu-layers all` or a lower layer count if your llama.cpp build supports it and your GPU
memory budget requires it. On CPU-only machines, omit `--n-gpu-layers`; generation will be slower but
the app still works.

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

JSON artifacts are validated before they are written. The final report records the analysis mode,
pose source, coach provider, coach model, verifier status, and downloadable artifact URLs.

## Project Structure

```text
app.py
web/
  index.html
  app.js
  report.js
  styles.css
src/pozify/
  pipeline.py
  contracts.py
  steps/
    video_qc.py
    pose_landmarker.py
    exercise_classifier.py
    annotated_renderer.py
    coach_summary.py
    verifier.py
  ml/
    exercise_router_features.py
    exercise_router_temporal.py
    exercise_router_inference.py
  slm/
    providers.py
    prompting.py
  exercises/
    squat/
    push_up/
    shoulder_press/
scripts/
  exercise_router_modal.py
  upload_exercise_router_to_hf.py
docs/
  build-small-hackathon-report.md
  exercise-router-training-report.md
  huggingface-router-release.md
  custom-data-collection-guide.md
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

## Notes And Limits

- Router metrics are from the current Modal feature cache, not a broad production benchmark.
- The app depends on full-body framing and usable pose extraction.
- Unsupported or low-confidence clips should route to `unknown` instead of receiving fabricated
  coaching.
- Qwen is prompted and verified; it is not fine-tuned in this project.
- More held-out videos, camera angles, body types, and custom unknown clips are needed before using
  Pozify as a robust training product.
