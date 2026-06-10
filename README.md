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

`annotated_video.mp4` is not implemented yet. The mocked renderer currently returns the original input video path and writes `annotated_video_placeholder.json`.

## Replacing Mock Steps

Each step lives in `src/pozify/steps/` and exposes a `run(...)` function.

Recommended replacement order:

1. `video_qc.py`: read real video metadata with OpenCV.
2. `pose_landmarker.py`: integrate MediaPipe Pose Landmarker.
3. `exercise_classifier.py`: load the exercise router model.
4. `rep_counter.py`: implement exercise-specific state machines.
5. `rep_analysis.py` and `issue_marker.py`: compute real metrics and issue scores.
6. `annotated_renderer.py`: render skeleton overlays and issue highlights.
7. `coach_summary.py`: call the selected small language model with retrieved knowledge cards.

## Development Checks

```bash
python3 -m py_compile app.py src/pozify/*.py src/pozify/steps/*.py
uv run python -c "import app; from pozify.pipeline import run_pipeline; print('ok')"
```
