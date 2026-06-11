from __future__ import annotations

from pozify.contracts import PoseSequence, UserProfile, VideoManifest
from pozify.exercises.base import ExerciseStrategy
from pozify.exercises.push_up import PushUpExercise
from pozify.exercises.shoulder_press import ShoulderPressExercise
from pozify.exercises.squat import SquatExercise
from pozify.exercises.unknown import UnknownExercise


EXERCISE_CLASSES: dict[str, type[ExerciseStrategy]] = {
    "push_up": PushUpExercise,
    "shoulder_press": ShoulderPressExercise,
    "squat": SquatExercise,
    "unknown": UnknownExercise,
}


def create_exercise_strategy(
    exercise: str,
    *,
    video_manifest: VideoManifest,
    pose_sequence: PoseSequence,
    profile: UserProfile,
) -> ExerciseStrategy:
    exercise_class = EXERCISE_CLASSES.get(exercise, UnknownExercise)
    return exercise_class(
        video_manifest=video_manifest,
        pose_sequence=pose_sequence,
        profile=profile,
    )
