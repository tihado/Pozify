from __future__ import annotations

from pozify.steps.exercise_analyzers.base import ExerciseAnalyzer, ExerciseMetricResult
from pozify.steps.exercise_analyzers.push_up import PushUpAnalyzer
from pozify.steps.exercise_analyzers.shoulder_press import ShoulderPressAnalyzer
from pozify.steps.exercise_analyzers.squat import SquatAnalyzer
from pozify.steps.exercise_analyzers.unknown import UnknownAnalyzer


def analyzer_for(exercise: str) -> ExerciseAnalyzer:
    analyzers: dict[str, ExerciseAnalyzer] = {
        "push_up": PushUpAnalyzer(),
        "shoulder_press": ShoulderPressAnalyzer(),
        "squat": SquatAnalyzer(),
    }
    return analyzers.get(exercise, UnknownAnalyzer())


__all__ = ["ExerciseAnalyzer", "ExerciseMetricResult", "analyzer_for"]
