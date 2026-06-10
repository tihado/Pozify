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

Setting `POZIFY_MOCK_MODE=0` enables real video QC and MediaPipe-backed pose extraction while later
analysis steps remain mocked. On current Python versions the pose step uses MediaPipe Tasks and
downloads `pose_landmarker_lite.task` into `/tmp/pozify-models` on first use. To use a pre-downloaded
model, set:

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
- `rep_analysis.json`
- `variation.json`
- `issue_markers.json`
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

`annotated_video.mp4` is not implemented yet. The mocked renderer currently returns the original input video path and writes `annotated_video_placeholder.json`.

## Replacing Mock Steps

Each step lives in `src/pozify/steps/` and exposes a `run(...)` function.

Recommended replacement order:

1. `video_qc.py`: read real video metadata with OpenCV.
2. `pose_backends/`: add or refine pose model adapters such as MediaPipe or MMPose.
3. `exercise_classifier.py`: load the exercise router model.
4. `rep_counter.py`: implement exercise-specific state machines.
5. `rep_analysis.py` and `issue_marker.py`: compute real metrics and issue scores.
6. `annotated_renderer.py`: render skeleton overlays and issue highlights.
7. `coach_summary.py`: call the selected small language model with retrieved knowledge cards.

## Development Checks

```bash
uv run python -m unittest discover -s tests
python3 -m py_compile app.py src/pozify/*.py src/pozify/steps/*.py
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
