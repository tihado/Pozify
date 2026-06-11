from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence, RepAnalysis, Reps
from pozify.exercise_catalog import get_exercise_spec
from pozify.exercises.base import ExerciseStrategy
from pozify.steps.exercise_analyzers.unknown import UnknownAnalyzer
from pozify.steps.rep_signals import SignalSample


class UnknownExercise(UnknownAnalyzer, ExerciseStrategy):
    exercise = "unknown"

    def build_signal(self, sequence: PoseSequence) -> tuple[list[SignalSample], dict[str, Any]]:
        return [], {"selected_signal": "none", "thresholds": {}, "extrema": [], "accepted_reps": []}

    def count(self, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
        reps = Reps(exercise=self.exercise, reps=[], partial_reps=[{"reason": "unknown_exercise"}])
        return reps, {"selected_signal": "none", "thresholds": {}, "extrema": [], "accepted_reps": []}

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        exercise_spec = get_exercise_spec(analysis.exercise)
        return (
            exercise_spec.default_variation,
            exercise_spec.default_variation_confidence,
            list(exercise_spec.default_variation_not_issues),
        )
