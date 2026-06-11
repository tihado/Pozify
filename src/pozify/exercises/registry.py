from __future__ import annotations

from pozify.exercises.base import ExerciseStrategy
from pozify.exercises.push_up import PushUpExercise
from pozify.exercises.shoulder_press import ShoulderPressExercise
from pozify.exercises.squat import SquatExercise
from pozify.exercises.unknown import UnknownExercise


EXERCISES: dict[str, ExerciseStrategy] = {
    "push_up": PushUpExercise(),
    "shoulder_press": ShoulderPressExercise(),
    "squat": SquatExercise(),
    "unknown": UnknownExercise(),
}


def get_exercise_strategy(exercise: str) -> ExerciseStrategy:
    return EXERCISES.get(exercise, EXERCISES["unknown"])
