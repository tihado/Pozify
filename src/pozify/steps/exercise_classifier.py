from __future__ import annotations

from pozify.contracts import ExerciseClassification, PoseSequence, UserProfile
from pozify.exercise_catalog import DEFAULT_AUTO_EXERCISE


def run(sequence: PoseSequence, profile: UserProfile) -> ExerciseClassification:
    if profile.intended_exercise != "auto":
        exercise = profile.intended_exercise
        confidence = 0.98
        fallback_required = False
    else:
        exercise = DEFAULT_AUTO_EXERCISE
        confidence = 0.92
        fallback_required = False

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
            for frame in sequence.frames[:4]
        ],
        fallback_required=fallback_required,
    )
