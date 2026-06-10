from __future__ import annotations

from typing import Any

from pozify.contracts import ExerciseClassification, PoseSequence, Reps
from pozify.steps.rep_counters import get_rep_counter


def run(classification: ExerciseClassification, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
    return get_rep_counter(classification.exercise).count(sequence)
