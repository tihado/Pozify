# Pozify Implementation Plan

This plan breaks Pozify into implementation workstreams that can be developed independently while preserving the current JSON data contracts.

## Phase 1: Foundation And Contracts

Goal: make the mocked pipeline production-shaped before replacing mock logic.

Deliverables:

- Stable JSON contracts in `src/pozify/contracts.py`.
- Artifact persistence under `runs/<run_id>/`.
- Contract validation and unit tests.
- Clear step boundaries for replacing mock implementations.
- Seed knowledge card structure.

## Phase 2: Video Intake And Quality Gate

Goal: accept user videos, extract reliable metadata, and reject videos that cannot be analyzed safely.

Deliverables:

- OpenCV-backed video metadata extraction.
- Duration, FPS, brightness, blur, and basic frame-count checks.
- Pose-quality-aware analysis gate.
- User-facing quality warnings.
- Capture guidance for failed videos.

## Phase 3: Pose Extraction And Cleaning

Goal: convert videos into a clean 17-point 3D pose sequence.

Deliverables:

- MediaPipe Pose Landmarker integration.
- Per-frame landmark persistence.
- Pose quality scoring.
- Smoothing and interpolation.
- Coordinate normalization.

## Phase 4: Exercise Router

Goal: classify the video as squat, push-up, shoulder press, or unknown.

Deliverables:

- Pose feature extraction for 30-frame windows.
- Baseline classifier.
- Fine-tuned small temporal classifier.
- Confidence-weighted aggregation.
- Manual fallback when confidence is low.

## Phase 5: Rep Segmentation

Goal: split each exercise into clean reps using transparent state machines.

Deliverables:

- Squat rep counter.
- Push-up rep counter.
- Shoulder press rep counter.
- Partial-rep handling.
- Rep boundary visualization/debug output.

## Phase 6: Rep Analysis And Variation Detection

Goal: compute meaningful metrics per rep and separate valid variations from issues.

Deliverables:

- Common rep metrics.
- Exercise-specific metrics.
- Variation detection rules.
- Aggregate trends such as fatigue and decreasing ROM.

## Phase 7: Frame-Level Issue Markers

Goal: identify exact issue intervals and attach evidence to each issue.

Deliverables:

- Rule-based frame-level issue scores.
- Consecutive-frame thresholding.
- Issue interval grouping.
- Optional temporal issue classifier.
- Evidence payloads for each marker.

## Phase 8: Annotated Video And UI

Goal: make the evidence visible and easy to inspect.

Deliverables:

- Skeleton overlay renderer.
- Red/amber highlights during issue intervals.
- Rep counter overlay.
- Issue thumbnails.
- Gradio report layout with summary, metrics, reps, issues, and JSON evidence tabs.

## Phase 9: Grounded Coach Summary

Goal: generate concise coaching feedback from structured evidence, not model memory.

Deliverables:

- Exercise, variation, issue, goal, and safety knowledge cards.
- Deterministic retrieval.
- SLM prompt contract.
- Summary verifier.
- Conservative fallback summary.

## Phase 10: Dataset, Training, Evaluation, And Demo

Goal: create a defensible small-model story and a reliable hackathon demo.

Deliverables:

- Dataset ingestion scripts.
- Custom data collection guide.
- Training pipeline for the exercise router.
- Evaluation metrics.
- Demo videos.
- README and deployment instructions.
