from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import ExerciseClassification, PoseFrame, PoseSequence
from pozify.steps import rep_counter


def _frame(frame_index: int, landmarks: dict[str, dict[str, float]]) -> PoseFrame:
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"mean_visibility": 0.95, "normalized": True},
    )


def _wave(frame_index: int, cycle_frames: int, *, phase_shift: float = 0.0) -> float:
    return (1.0 - math.cos(2.0 * math.pi * ((frame_index + phase_shift) / cycle_frames))) / 2.0


def _squat_landmarks(depth: float) -> dict[str, dict[str, float]]:
    hip_y = 0.52 + depth * 0.18
    shoulder_y = 0.28 + depth * 0.06
    knee_y = 0.7
    ankle_y = 0.92
    hip_x_left, hip_x_right = 0.43, 0.57
    knee_x_left = hip_x_left + depth * 0.06
    knee_x_right = hip_x_right - depth * 0.06
    return {
        "left_shoulder": {"x": 0.4, "y": shoulder_y, "smoothed_x": 0.4, "smoothed_y": shoulder_y},
        "right_shoulder": {"x": 0.6, "y": shoulder_y, "smoothed_x": 0.6, "smoothed_y": shoulder_y},
        "left_hip": {"x": hip_x_left, "y": hip_y, "smoothed_x": hip_x_left, "smoothed_y": hip_y},
        "right_hip": {"x": hip_x_right, "y": hip_y, "smoothed_x": hip_x_right, "smoothed_y": hip_y},
        "left_knee": {"x": knee_x_left, "y": knee_y, "smoothed_x": knee_x_left, "smoothed_y": knee_y},
        "right_knee": {"x": knee_x_right, "y": knee_y, "smoothed_x": knee_x_right, "smoothed_y": knee_y},
        "left_ankle": {"x": 0.42, "y": ankle_y, "smoothed_x": 0.42, "smoothed_y": ankle_y},
        "right_ankle": {"x": 0.58, "y": ankle_y, "smoothed_x": 0.58, "smoothed_y": ankle_y},
    }


def _push_up_landmarks(depth: float) -> dict[str, dict[str, float]]:
    shoulder_y = 0.38 + depth * 0.16
    hip_y = 0.46 + depth * 0.16
    ankle_y = 0.54 + depth * 0.16
    elbow_y = 0.46 + depth * 0.1
    wrist_y = 0.5 + depth * 0.04
    elbow_x_left = 0.36 + depth * 0.02
    elbow_x_right = 0.64 - depth * 0.02
    return {
        "left_shoulder": {"x": 0.3, "y": shoulder_y, "smoothed_x": 0.3, "smoothed_y": shoulder_y},
        "right_shoulder": {"x": 0.7, "y": shoulder_y, "smoothed_x": 0.7, "smoothed_y": shoulder_y},
        "left_elbow": {"x": elbow_x_left, "y": elbow_y, "smoothed_x": elbow_x_left, "smoothed_y": elbow_y},
        "right_elbow": {"x": elbow_x_right, "y": elbow_y, "smoothed_x": elbow_x_right, "smoothed_y": elbow_y},
        "left_wrist": {"x": 0.34, "y": wrist_y, "smoothed_x": 0.34, "smoothed_y": wrist_y},
        "right_wrist": {"x": 0.66, "y": wrist_y, "smoothed_x": 0.66, "smoothed_y": wrist_y},
        "left_hip": {"x": 0.42, "y": hip_y, "smoothed_x": 0.42, "smoothed_y": hip_y},
        "right_hip": {"x": 0.58, "y": hip_y, "smoothed_x": 0.58, "smoothed_y": hip_y},
        "left_ankle": {"x": 0.44, "y": ankle_y, "smoothed_x": 0.44, "smoothed_y": ankle_y},
        "right_ankle": {"x": 0.56, "y": ankle_y, "smoothed_x": 0.56, "smoothed_y": ankle_y},
    }


def _shoulder_press_landmarks(lift: float) -> dict[str, dict[str, float]]:
    wrist_y = 0.76 - lift * 0.34
    elbow_y = 0.6 - lift * 0.1
    elbow_x_left = 0.4 - lift * 0.05
    elbow_x_right = 0.6 + lift * 0.05
    return {
        "left_shoulder": {"x": 0.42, "y": 0.42, "smoothed_x": 0.42, "smoothed_y": 0.42},
        "right_shoulder": {"x": 0.58, "y": 0.42, "smoothed_x": 0.58, "smoothed_y": 0.42},
        "left_elbow": {"x": elbow_x_left, "y": elbow_y, "smoothed_x": elbow_x_left, "smoothed_y": elbow_y},
        "right_elbow": {"x": elbow_x_right, "y": elbow_y, "smoothed_x": elbow_x_right, "smoothed_y": elbow_y},
        "left_wrist": {"x": 0.4, "y": wrist_y, "smoothed_x": 0.4, "smoothed_y": wrist_y},
        "right_wrist": {"x": 0.6, "y": wrist_y, "smoothed_x": 0.6, "smoothed_y": wrist_y},
    }


def _sequence_for_exercise(exercise: str, cycles: int, *, partial_tail: bool = False) -> PoseSequence:
    cycle_frames = 24
    total_frames = cycles * cycle_frames + (cycle_frames // 2 if partial_tail else 0)
    frames: list[PoseFrame] = []
    for frame_index in range(total_frames):
        wave = _wave(frame_index, cycle_frames)
        if exercise == "squat":
            landmarks = _squat_landmarks(wave)
        elif exercise == "shoulder_press":
            landmarks = _shoulder_press_landmarks(wave)
        else:
            landmarks = _push_up_landmarks(wave)
        frames.append(_frame(frame_index, landmarks))
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0,
    )


class RepCounterTests(unittest.TestCase):
    def _classification(self, exercise: str) -> ExerciseClassification:
        return ExerciseClassification(
            exercise=exercise,  # type: ignore[arg-type]
            confidence=0.95,
            window_predictions=[],
            fallback_required=False,
        )

    def test_segments_squat_reps(self) -> None:
        reps, debug = rep_counter.run(self._classification("squat"), _sequence_for_exercise("squat", 3))
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.reps[0].start_frame, 0)
        self.assertTrue(debug["accepted_reps"])

    def test_segments_push_up_reps(self) -> None:
        reps, _debug = rep_counter.run(self._classification("push_up"), _sequence_for_exercise("push_up", 3))
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.partial_reps, [])

    def test_segments_shoulder_press_reps(self) -> None:
        reps, _debug = rep_counter.run(
            self._classification("shoulder_press"),
            _sequence_for_exercise("shoulder_press", 3),
        )
        self.assertEqual(len(reps.reps), 3)
        self.assertEqual(reps.reps[0].mid_frame, 12)

    def test_reports_partial_last_rep(self) -> None:
        reps, _debug = rep_counter.run(
            self._classification("push_up"),
            _sequence_for_exercise("push_up", 2, partial_tail=True),
        )
        self.assertEqual(len(reps.reps), 2)
        self.assertIn("ends_mid_rep", {item["reason"] for item in reps.partial_reps})

    def test_unknown_exercise_is_not_segmented(self) -> None:
        reps, debug = rep_counter.run(self._classification("unknown"), _sequence_for_exercise("push_up", 1))
        self.assertEqual(reps.reps, [])
        self.assertEqual(reps.partial_reps, [{"reason": "unknown_exercise"}])
        self.assertEqual(debug["selected_signal"], "none")


if __name__ == "__main__":
    unittest.main()
