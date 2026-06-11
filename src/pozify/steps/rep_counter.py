from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence, Reps
from pozify.exercises import ExerciseStrategy


def run(exercise: ExerciseStrategy, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
    return exercise.count(sequence)
