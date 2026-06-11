from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import (
    ExerciseClassification,
    PoseFrame,
    PoseSequence,
    Rep,
    Reps,
    UserProfile,
)
from pozify.steps import rep_analysis, variation_detector


def _frame(frame_index: int, landmarks: dict[str, dict[str, float]]) -> PoseFrame:
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"mean_visibility": 0.95, "normalized": True},
    )


def _wave(frame_index: int, cycle_frames: int) -> float:
    return (1.0 - math.cos(2.0 * math.pi * (frame_index / cycle_frames))) / 2.0


def _push_up_landmarks(depth: float, *, hand_ratio: float = 1.7) -> dict[str, dict[str, float]]:
    shoulder_y = 0.38 + depth * 0.16
    hip_y = 0.46 + depth * 0.16
    ankle_y = 0.54 + depth * 0.16
    elbow_y = 0.46 + depth * 0.1
    wrist_y = 0.5 + depth * 0.04
    shoulder_width = 0.4
    hand_width = shoulder_width * hand_ratio
    return {
        "left_shoulder": {"x": 0.3, "y": shoulder_y},
        "right_shoulder": {"x": 0.7, "y": shoulder_y},
        "left_elbow": {"x": 0.36 + depth * 0.02, "y": elbow_y},
        "right_elbow": {"x": 0.64 - depth * 0.02, "y": elbow_y},
        "left_wrist": {"x": 0.5 - hand_width / 2, "y": wrist_y},
        "right_wrist": {"x": 0.5 + hand_width / 2, "y": wrist_y},
        "left_hip": {"x": 0.42, "y": hip_y},
        "right_hip": {"x": 0.58, "y": hip_y},
        "left_ankle": {"x": 0.44, "y": ankle_y},
        "right_ankle": {"x": 0.56, "y": ankle_y},
    }


def _squat_landmarks(depth: float, *, stance_ratio: float = 1.5) -> dict[str, dict[str, float]]:
    shoulder_width = 0.2
    stance_width = shoulder_width * stance_ratio
    hip_y = 0.52 + depth * 0.22
    shoulder_y = 0.28 + depth * 0.06
    knee_y = 0.7
    return {
        "left_shoulder": {"x": 0.4, "y": shoulder_y},
        "right_shoulder": {"x": 0.6, "y": shoulder_y},
        "left_hip": {"x": 0.43, "y": hip_y},
        "right_hip": {"x": 0.57, "y": hip_y},
        "left_knee": {"x": 0.43 + depth * 0.05, "y": knee_y},
        "right_knee": {"x": 0.57 - depth * 0.05, "y": knee_y},
        "left_ankle": {"x": 0.5 - stance_width / 2, "y": 0.92},
        "right_ankle": {"x": 0.5 + stance_width / 2, "y": 0.92},
    }


def _shoulder_press_landmarks(lift: float, *, partial: bool = True) -> dict[str, dict[str, float]]:
    wrist_top = 0.56 if partial else 0.42
    wrist_y = 0.76 - lift * (0.76 - wrist_top)
    elbow_y = 0.6 - lift * 0.06
    return {
        "left_shoulder": {"x": 0.42, "y": 0.42},
        "right_shoulder": {"x": 0.58, "y": 0.42},
        "left_elbow": {"x": 0.4 - lift * 0.02, "y": elbow_y},
        "right_elbow": {"x": 0.6 + lift * 0.02, "y": elbow_y},
        "left_wrist": {"x": 0.4, "y": wrist_y},
        "right_wrist": {"x": 0.6, "y": wrist_y},
        "left_hip": {"x": 0.44, "y": 0.74},
        "right_hip": {"x": 0.56, "y": 0.74},
    }


def _sequence(exercise: str) -> PoseSequence:
    frames = []
    for frame_index in range(25):
        wave = _wave(frame_index, 24)
        if exercise == "push_up":
            landmarks = _push_up_landmarks(wave)
        elif exercise == "shoulder_press":
            landmarks = _shoulder_press_landmarks(wave)
        else:
            landmarks = _squat_landmarks(wave)
        frames.append(_frame(frame_index, landmarks))
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="none",
        pose_valid_ratio=1.0,
    )


def _classification(exercise: str) -> ExerciseClassification:
    return ExerciseClassification(
        exercise=exercise,
        confidence=0.95,
        window_predictions=[],
        fallback_required=False,
    )


def _reps(exercise: str) -> Reps:
    return Reps(
        exercise=exercise,
        reps=[Rep(1, 0, 12, 24, 0.0, 0.4, 0.8)],
        partial_reps=[],
    )


def _profile() -> UserProfile:
    return UserProfile(
        goal="beginner_practice",
        experience_level="beginner",
        intended_exercise="auto",
        intended_variation=None,
        known_limitations=[],
        equipment="bodyweight",
    )


class RepAnalysisVariationTests(unittest.TestCase):
    def test_push_up_metrics_detect_wide_grip_variation(self) -> None:
        analysis = rep_analysis.run(
            _classification("push_up"),
            _reps("push_up"),
            _sequence("push_up"),
        )
        variation = variation_detector.run(_classification("push_up"), analysis, _profile())

        self.assertEqual(variation.detected_variation, "wide_grip_push_up")
        self.assertIn("wide_hand_placement", variation.not_issues)
        self.assertGreater(analysis.items[0].metrics["hand_width_ratio"], 1.45)
        self.assertIn("body_line_score", analysis.items[0].metrics)

    def test_squat_metrics_detect_wide_stance_variation(self) -> None:
        analysis = rep_analysis.run(_classification("squat"), _reps("squat"), _sequence("squat"))
        variation = variation_detector.run(_classification("squat"), analysis, _profile())

        self.assertEqual(variation.detected_variation, "wide_squat_stance")
        self.assertIn("wide_stance", variation.not_issues)
        self.assertGreater(analysis.items[0].metrics["stance_width_ratio"], 1.35)
        self.assertIn("min_knee_angle_deg", analysis.items[0].metrics)

    def test_shoulder_press_metrics_detect_partial_press_variation(self) -> None:
        analysis = rep_analysis.run(
            _classification("shoulder_press"),
            _reps("shoulder_press"),
            _sequence("shoulder_press"),
        )
        variation = variation_detector.run(_classification("shoulder_press"), analysis, _profile())

        self.assertEqual(variation.detected_variation, "partial_press")
        self.assertIn("partial_range_of_motion", variation.not_issues)
        self.assertIn("lockout_quality", analysis.items[0].metrics)
        self.assertLess(analysis.aggregate_metrics["avg_wrist_travel"], 0.24)

    def test_profile_intended_variation_overrides_metric_rule(self) -> None:
        profile = UserProfile(
            goal="beginner_practice",
            experience_level="beginner",
            intended_exercise="push_up",
            intended_variation="close_grip_push_up",
            known_limitations=[],
            equipment="bodyweight",
        )
        analysis = rep_analysis.run(
            _classification("push_up"),
            _reps("push_up"),
            _sequence("push_up"),
        )
        variation = variation_detector.run(_classification("push_up"), analysis, profile)

        self.assertEqual(variation.detected_variation, "close_grip_push_up")
        self.assertEqual(variation.variation_confidence, 0.95)
        self.assertIn("close_hand_placement", variation.not_issues)


if __name__ == "__main__":
    unittest.main()
