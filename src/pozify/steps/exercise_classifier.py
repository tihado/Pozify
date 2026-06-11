from __future__ import annotations

from pathlib import Path

from pozify.contracts import ExerciseClassification, PoseSequence, UserProfile
from pozify.exercise_catalog import DEFAULT_AUTO_EXERCISE
from pozify.hf_spaces import default_spaces_gpu_duration, spaces_gpu
from pozify.ml.exercise_router_features import RouterWindow, extract_router_windows
from pozify.ml.exercise_router_inference import (
    MIN_POSE_VALID_RATIO,
    RouterModelBundle,
    aggregate_window_predictions,
    contract_window_predictions,
    load_router_model,
    predict_window_probabilities,
    window_predictions_from_scores,
)


def _sample_prediction_frames(sequence: PoseSequence, count: int = 4) -> list[object]:
    if not sequence.frames:
        return []
    if len(sequence.frames) <= count:
        return sequence.frames
    last_index = len(sequence.frames) - 1
    positions = sorted({round(index * last_index / (count - 1)) for index in range(count)})
    return [sequence.frames[position] for position in positions]


def _fixed_classification(
    sequence: PoseSequence,
    *,
    exercise: str,
    confidence: float,
    fallback_required: bool,
) -> ExerciseClassification:
    return ExerciseClassification(
        exercise=exercise,  # type: ignore[arg-type]
        confidence=confidence,
        window_predictions=[
            {
                "start_sec": frame.timestamp_sec,
                "end_sec": round(frame.timestamp_sec + 1.0, 3),
                "label": exercise,
                "confidence": confidence,
            }
            for frame in _sample_prediction_frames(sequence)
        ],
        fallback_required=fallback_required,
    )


def _manual_classification(sequence: PoseSequence, exercise: str) -> ExerciseClassification:
    windows = extract_router_windows(sequence, min_mean_visibility=0.0)
    if not windows:
        return _fixed_classification(
            sequence,
            exercise=exercise,
            confidence=0.98,
            fallback_required=False,
        )
    return ExerciseClassification(
        exercise=exercise,  # type: ignore[arg-type]
        confidence=0.98,
        window_predictions=[
            {
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "label": exercise,
                "confidence": 0.98,
            }
            for window in windows
        ],
        fallback_required=False,
    )


def _unknown_fallback(
    window_predictions: list[dict[str, float | str]] | None = None,
    *,
    confidence: float = 0.0,
) -> ExerciseClassification:
    return ExerciseClassification(
        exercise="unknown",
        confidence=round(confidence, 4),
        window_predictions=window_predictions or [],
        fallback_required=True,
    )


def _gpu_duration(*_args: object, **_kwargs: object) -> int:
    return default_spaces_gpu_duration()


@spaces_gpu(duration=_gpu_duration)
def _predict_router_scores(
    windows: list[RouterWindow],
    model_dir: str,
) -> list[dict[str, float]]:
    bundle = load_router_model(Path(model_dir))
    if bundle is None:
        return []
    return predict_window_probabilities(bundle, windows)


def run(
    sequence: PoseSequence,
    profile: UserProfile,
    *,
    mock: bool = False,
    model_bundle: RouterModelBundle | None = None,
    model_dir: Path | None = None,
) -> ExerciseClassification:
    if profile.intended_exercise != "auto":
        return _manual_classification(sequence, profile.intended_exercise)

    if mock:
        return _fixed_classification(
            sequence,
            exercise=DEFAULT_AUTO_EXERCISE,
            confidence=0.92,
            fallback_required=False,
        )

    windows = extract_router_windows(sequence)
    if not windows or sequence.pose_valid_ratio < MIN_POSE_VALID_RATIO:
        return _unknown_fallback()

    try:
        if model_bundle is not None:
            score_rows = predict_window_probabilities(model_bundle, windows)
        else:
            score_rows = _predict_router_scores(
                windows,
                str(model_dir or Path("models/exercise_router/active")),
            )
            if not score_rows:
                return _unknown_fallback()
    except Exception:
        return _unknown_fallback()

    predictions = window_predictions_from_scores(windows, score_rows)
    window_payload = contract_window_predictions(predictions)
    aggregated = aggregate_window_predictions(predictions)
    return ExerciseClassification(
        exercise=aggregated.label,  # type: ignore[arg-type]
        confidence=aggregated.confidence,
        window_predictions=window_payload,
        fallback_required=aggregated.fallback_required,
    )
