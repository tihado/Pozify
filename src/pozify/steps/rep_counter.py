from __future__ import annotations

from typing import Any

from pozify.contracts import ExerciseClassification, PoseSequence, Reps
from pozify.exercises import get_exercise_strategy


def run(classification: ExerciseClassification, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
    return get_exercise_strategy(classification.exercise).count(sequence)
