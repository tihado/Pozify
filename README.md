---
title: Pozify
emoji: 🏋️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.17.3"
python_version: "3.13"
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

The current implementation contains the full application pipeline, step inputs/outputs, and JSON data contracts. The model and computer-vision steps are intentionally mocked so the team can replace each step with real implementations without changing the pipeline interface.

## Run The App

```bash
uv run python app.py
```

You can also run:

```bash
uv run gradio app.py
```

The app runs at `http://127.0.0.1:7860` by default.

The pipeline runs in mock mode by default. To be explicit in scripts, call
`run_pipeline(..., mock=True)` or set:

```bash
POZIFY_MOCK_MODE=1 uv run python app.py
```

To run the end-to-end app with real video QC, MediaPipe pose extraction, real rep segmentation, and
annotated video rendering, set:

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

## Project Structure

```text
app.py
src/pozify/
  contracts.py
  pipeline.py
  artifacts.py
  steps/
    video_qc.py
    pose_landmarker.py
    pose_cleaning.py
    exercise_classifier.py
    rep_counter.py
    rep_analysis.py
    variation_detector.py
    issue_marker.py
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

`annotated_video.mp4` is produced when the renderer can decode the input video. It overlays pose
landmarks, skeleton connections, rep count, and rep boundary labels on top of the source video.

## Replacing Mock Steps

Each step lives in `src/pozify/steps/` and exposes a `run(...)` function.

## Adding Exercises

Supported exercise metadata is centralized in `src/pozify/exercise_catalog.py`. To add a new mocked
exercise, add one `ExerciseSpec` with:

- `key` and `display_name`
- `default_variation` and confidence
- `metric_factory` for mock per-rep metrics
- optional `variation_hints`, default `not_issues`, and `mock_issue`

The Gradio dropdown, profile validation, variation detector, rep analysis, and issue marker read from
this catalog, so new exercises do not require changing each step just to become selectable. Real
exercise-specific algorithms can still replace individual step implementations later while preserving
the same pipeline contracts.

Recommended replacement order:

1. `video_qc.py`: read real video metadata with OpenCV.
2. `pose_backends/`: add or refine pose model adapters such as MediaPipe or MMPose.
3. `exercise_classifier.py`: load the exercise router model.
4. `rep_counter.py`: implement exercise-specific state machines.
5. `rep_analysis.py` and `issue_marker.py`: compute real metrics and issue scores.
6. `annotated_renderer.py`: render skeleton overlays and issue highlights.
7. `coach_summary.py`: call the selected small language model with retrieved knowledge cards.

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
Baseline wins ties to keep runtime inference lightweight unless the temporal model clearly improves
the metrics.

Download the selected artifact and its selection file after evaluation, then place them under:

```text
models/exercise_router/active/
```

For the baseline this directory should contain `router.joblib`; for the temporal model it should
contain `router.pt`. In both cases, include `router_selection.json` when present so the local loader
can choose the intended active artifact even if multiple artifacts exist.

Custom unknown clips can be uploaded into the data volume at `/data/raw/custom_unknown/` before the
`features` stage. Use consented clips only; useful unknown examples include idle standing, walking
into frame, setup motion, stretching, bad camera angle, and partial/unsupported reps. Unsupported
classes from the Riccio dataset, such as bicep curl, are mapped to `unknown`.

## Development Checks

```bash
uv run python -m unittest discover -s tests
python3 -m py_compile app.py src/pozify/*.py src/pozify/steps/*.py src/pozify/ml/*.py scripts/*.py
uv run python -c "import app; from pozify.pipeline import run_pipeline; print('ok')"
```

The unit tests run the full mocked pipeline with no video input and with a small fixture path, then
assert deterministic top-level keys for each JSON artifact.

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
