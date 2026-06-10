from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence, Reps
from pozify.steps.rep_counters.base import ExerciseRepCounter
from pozify.steps.rep_counters.push_up import PushUpRepCounter
from pozify.steps.rep_counters.shoulder_press import ShoulderPressRepCounter
from pozify.steps.rep_counters.squat import SquatRepCounter
from pozify.steps.rep_signals import SignalSample


class UnknownRepCounter(ExerciseRepCounter):
    exercise = "unknown"

    def build_signal(self, sequence: PoseSequence) -> tuple[list[SignalSample], dict[str, Any]]:
        return [], {"selected_signal": "none", "thresholds": {}, "extrema": [], "accepted_reps": []}

    def count(self, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
        reps = Reps(exercise=self.exercise, reps=[], partial_reps=[{"reason": "unknown_exercise"}])
        return reps, {"selected_signal": "none", "thresholds": {}, "extrema": [], "accepted_reps": []}


REP_COUNTERS: dict[str, ExerciseRepCounter] = {
    "push_up": PushUpRepCounter(),
    "shoulder_press": ShoulderPressRepCounter(),
    "squat": SquatRepCounter(),
    "unknown": UnknownRepCounter(),
}


def get_rep_counter(exercise: str) -> ExerciseRepCounter:
    return REP_COUNTERS.get(exercise, REP_COUNTERS["unknown"])

