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
short_description: Video-based workout form review app.
tags:
  - gradio
  - computer-vision
  - pose-estimation
  - fitness
  - video-analysis
---

# Pozify

Pozify is a Gradio + uv base project for a video-based workout form review app.

The current implementation contains the full application pipeline, step inputs/outputs, and JSON data contracts. Uploaded videos use real video QC, MediaPipe pose extraction, and real rep segmentation by default; no-video/demo runs still use mock pose data.

## Run The App

```bash
uv run python app.py
```

You can also run:

```bash
uv run gradio app.py
```

The app runs at `http://127.0.0.1:7860` by default.

For in-browser playback of `annotated_video.mp4` and issue clips, the machine should have `ffmpeg`
or another H.264-capable encoder available. Without that, Pozify can still render thumbnails and
analysis artifacts, but the UI will mark video playback as unavailable instead of showing a broken
player.

The pipeline runs with real pose extraction by default when a video path is provided. No-video runs
default to mock mode. To force mock mode in scripts, call `run_pipeline(..., mock=True)` or set:

```bash
POZIFY_MOCK_MODE=1 uv run python app.py
```

To force the end-to-end app to use real video QC, MediaPipe pose extraction, real rep segmentation,
and annotated video rendering even when `POZIFY_MOCK_MODE` is set elsewhere, set:

```bash
POZIFY_MOCK_MODE=0 uv run python app.py
```

On current Python versions the pose step uses MediaPipe Tasks and downloads
`pose_landmarker_lite.task` into `/tmp/pozify-models` on first use. To use a pre-downloaded model, set:

```bash
POZIFY_MEDIAPIPE_POSE_MODEL=/path/to/pose_landmarker_lite.task POZIFY_MOCK_MODE=0 uv run python app.py
```

The pose extractor is selected through a backend interface. MediaPipe is the default real backend:

```bash
POZIFY_POSE_BACKEND=mediapipe POZIFY_MOCK_MODE=0 uv run python app.py
```

## Open-Source SLM Summary Provider

The summary stack now supports an opt-in local open-source SLM provider behind the existing
provider/verifier/fallback flow. The default summary provider remains `template`.

To enable a real local summary model, install the optional dependencies first:

```bash
uv sync --extra summary
```

This installs the Python-side runtime needed for the local SLM path, including `transformers`.
The repo already includes `torch` in the base dependencies. The first run may still download the
configured model weights from Hugging Face if they are not already present in the local cache.

Then run the app with the local SLM provider enabled:

```bash
POZIFY_MOCK_MODE=0 \
POZIFY_SUMMARY_PROVIDER=slm_local \
POZIFY_SUMMARY_BACKEND=transformers \
POZIFY_SUMMARY_MODEL=Qwen/Qwen2.5-3B-Instruct \
uv run python app.py
```

The model output is still treated as a draft. It must parse as valid JSON and pass the verifier.
If parsing or verification fails, Pozify automatically falls back to the conservative template-based
summary path.

Relevant summary environment variables:

- `POZIFY_SUMMARY_PROVIDER=template|mock|unsafe_mock|slm_local`
- `POZIFY_SUMMARY_BACKEND=transformers`
- `POZIFY_SUMMARY_MODEL=Qwen/Qwen2.5-3B-Instruct`
- `POZIFY_SUMMARY_DEVICE=cpu|mps|cuda|auto`
- `POZIFY_SUMMARY_MAX_TOKENS=512`
- `POZIFY_SUMMARY_TEMPERATURE=0.2`

For stability on Apple Silicon, the local summary backend defaults to `POZIFY_SUMMARY_DEVICE=cpu`.
This avoids common MPS out-of-memory failures during summary generation. If you explicitly want to
try Metal acceleration, set `POZIFY_SUMMARY_DEVICE=mps`.

To verify that the run actually used the SLM provider, inspect the `JSON` tab or
`summary_generation.json` and confirm:

- `summary_provider` is `slm_local`
- `summary_backend` is `transformers`
- `summary_model` is `Qwen/Qwen2.5-3B-Instruct` or your configured model

If you instead see:

```json
"summary_provider": "template",
"summary_backend": null,
"summary_model": null
```

then the app did not receive `POZIFY_SUMMARY_PROVIDER=slm_local` and stayed on the default
template provider.

Common fixes:

1. Stop the running app process completely.
2. Start it again from the same terminal with the full command above.
3. Ensure the optional dependencies are installed with `uv sync --extra summary`.
4. Re-run the analysis and check `summary_generation.json` again.

Backend implementations live in `src/pozify/steps/pose_backends/` and return the same
`PoseDetection` shape, so downstream steps do not depend on a specific model library. A reserved
`mmpose` backend class is included as the integration point for OpenMMLab MMPose; implementing it
requires installing MMPose/MMCV and mapping model keypoints into the shared landmark dictionary.

When running in real mode, the UI summary now shows:

- `Analysis mode`: `mock` or `real`
- `Pose source`: for example `mock_pose` or `mediapipe_pose`

## Pipeline

```text
user profile + input video
-> video QC
-> 33-point pose landmarker
-> pose cleaning and normalization
-> exercise classifier
-> exercise-specific rep counter
-> per-rep analysis
-> variation detection
-> frame-level issue markers
-> annotated video renderer
-> grounded coach summary
-> verifier
-> final report
```

After classification, the pipeline creates one object for the detected exercise class with the video
manifest, cleaned pose sequence, and user profile. Rep counting, rep analysis, variation detection,
and issue marking then run as methods on that exercise object.

## Project Structure

```text
app.py
src/pozify/
  contracts.py
  pipeline.py
  artifacts.py
  exercises/
    push_up/
      strategy.py
      analyzer.py
      issue_markers.py
    squat/
      strategy.py
      analyzer.py
      issue_markers.py
    shoulder_press/
      strategy.py
      analyzer.py
      issue_markers.py
    shared/
      analyzer.py
      issue_marker.py
      rep_counter.py
  steps/
    video_qc.py
    pose_landmarker.py
    pose_cleaning.py
    exercise_classifier.py
    annotated_renderer.py
    coach_summary.py
    verifier.py
docs/
  pozify-project-documentation.md
```

## Generated Artifacts

Each analysis run creates a folder under `runs/<run_id>/`:

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
- `summary_generation.json`
- `verification.json`
- `final_report.json`

`manifest.json` indexes the generated artifacts in pipeline order and records whether the run used
mock mode. JSON artifacts are validated before they are written, including required fields, supported
enum values, score ranges, frame/timestamp ordering, and final report shape.

`video_manifest.json` is produced from OpenCV metadata and sampled-frame quality checks. It includes
FPS, duration, frame count, sampled frame count, resolution, codec/container when available,
brightness, blur score, warning labels, and `analysis_allowed`.

Video quality warning labels:

- `too_short`
- `too_long`
- `too_dark`
- `too_blurry`
- `fps_too_low`
- `resolution_too_low`
- `video_decode_failed`

Decode failures set `analysis_allowed=false`; the Gradio UI surfaces capture guidance instead of
coach-style feedback when analysis is blocked.

`rep_debug.json` stores the selected segmentation signal, thresholds, detected extrema, and accepted
rep segments for debugging rep counting.

`summary_generation.json` stores the summary provider name, backend name, model name, JSON parse
status, verifier result, and whether the conservative fallback summary was used.

`annotated_video.mp4` is produced when the renderer can decode the input video. It overlays pose
landmarks, skeleton connections, rep count, and rep boundary labels on top of the source video.

## Replacing Mock Steps

Each step lives in `src/pozify/steps/` and exposes a `run(...)` function.

## Adding Exercises

Supported exercise metadata is centralized in `src/pozify/exercise_catalog.py`. To add a new
exercise, add one `ExerciseSpec` with:

- `key` and `display_name`
- `default_variation` and confidence
- `metric_factory` for legacy/mock fallback metadata
- optional `variation_hints`, default `not_issues`, and `mock_issue`

The Gradio dropdown, profile validation, variation detector, rep analysis, and issue marker read from
this catalog, so new exercises do not require changing each step just to become selectable. Real
exercise-specific code lives under `src/pozify/exercises/<exercise>/`:

- `strategy.py`: rep-counting signal, variation logic, and exercise strategy wiring
- `analyzer.py`: per-rep metrics
- `issue_markers.py`: frame-level issue rules

Shared analyzer and issue marker primitives live under `src/pozify/exercises/shared/`.

Recommended replacement order:

1. `video_qc.py`: read real video metadata with OpenCV.
2. `pose_backends/`: add or refine pose model adapters such as MediaPipe or MMPose.
3. `exercise_classifier.py`: load the exercise router model.
4. `exercises/<exercise>/strategy.py`: implement exercise-specific rep signals and variation logic.
5. `exercises/<exercise>/analyzer.py`: compute per-rep metrics.
6. `exercises/<exercise>/issue_markers.py`: compute real issue scores and intervals.
7. `annotated_renderer.py`: render skeleton overlays and issue highlights.
8. `coach_summary.py`: call the selected small language model with retrieved knowledge cards.

## Exercise Router Training

Issue 4 adds a real exercise router path for `squat`, `push_up`, `shoulder_press`, and `unknown`.
Runtime inference stays local: `src/pozify/steps/exercise_classifier.py` looks for a trained artifact
under `models/exercise_router/active/` and falls back to `unknown` with `fallback_required=true`
when no model is present or confidence is too low.

Modal is used for dataset ingestion, batch feature extraction, and training:

```bash
uv run modal setup
uv run modal run scripts/exercise_router_modal.py --stage ingest
uv run modal run scripts/exercise_router_modal.py --stage features
uv run modal run scripts/exercise_router_modal.py --stage train-baseline
uv run modal run scripts/exercise_router_modal.py --stage train-temporal
uv run modal run scripts/exercise_router_modal.py --stage evaluate
```

The latest training metrics and selected-artifact result are recorded in
[docs/exercise-router-training-report.md](docs/exercise-router-training-report.md).
Custom data collection is documented in
[docs/custom-data-collection-guide.md](docs/custom-data-collection-guide.md). Demo clips are listed
in [demo/README.md](demo/README.md).

The Modal app uses:

- `pozify-router-data` for raw videos, manifests, and feature caches.
- `pozify-router-models` for `baseline.joblib`, `temporal.pt`, `evaluation.json`, and the selected
  router artifact.

The baseline trains a scikit-learn model over engineered window vectors. The temporal stage trains a
compact BiLSTM over 30-frame feature tensors on a Modal A10 GPU and writes `temporal.pt` plus
`temporal_metrics.json`. Its default hyperparameters follow the Riccio exercise-classification paper:
73 hidden units, 0.2174 dropout, 0.0004 learning rate, batch size 54, and 73 epochs
(https://arxiv.org/abs/2411.11548). Evaluation scores every available trained artifact, writes
per-model metrics into `evaluation.json`, and records the active artifact in `router_selection.json`.
The current selection policy prefers the BiLSTM temporal model when it is available; the baseline is
kept as a fallback/reference artifact.

Download the selected artifact and its selection file after evaluation, then place them under:

```text
models/exercise_router/active/
```

For the active BiLSTM router this directory should contain `temporal.pt` and
`router_selection.json`. Keep `router.joblib` only when you want the baseline artifact available for
comparison or fallback.

To publish and load the router from Hugging Face, use the setup notes in
[docs/huggingface-router-release.md](docs/huggingface-router-release.md). The draft model card is in
[docs/huggingface-router-model-card.md](docs/huggingface-router-model-card.md). Runtime loading uses
`build-small-hackathon/pozify-exercise-router` by default; set `POZIFY_ROUTER_HF_REPO_ID` only to
override it.

Custom unknown clips can be uploaded into the data volume at `/data/raw/custom_unknown/` before the
`features` stage. Use consented clips only; useful unknown examples include idle standing, walking
into frame, setup motion, stretching, bad camera angle, and partial/unsupported reps. Unsupported
classes from the Riccio dataset, such as bicep curl, are mapped to `unknown`.

## Development Checks

```bash
uv run python -m unittest discover -s tests
python3 -m py_compile app.py src/pozify/*.py src/pozify/steps/*.py src/pozify/exercises/*.py src/pozify/exercises/*/*.py
uv run python -c "import app; from pozify.pipeline import run_pipeline; print('ok')"
```

The unit tests run the full mocked pipeline explicitly, then assert deterministic top-level keys for
each JSON artifact.

## Git Hooks

Install Lefthook once per clone:

```bash
lefthook install
```

The configured `pre-commit` and `pre-push` hooks run:

```bash
uv run ruff check
uv run python -m unittest discover -s tests
```

To run the real MediaPipe smoke test against `tests/fixtures/sample.MOV`, opt in explicitly:

```bash
POZIFY_RUN_REAL_POSE_TESTS=1 uv run python -m unittest tests.test_pose_steps.PoseStepTests.test_real_sample_mov_extracts_pose_landmarks
```
