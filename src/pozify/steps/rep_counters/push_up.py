from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence
from pozify.steps.rep_counters.base import ExerciseRepCounter, combine, mean_optional, normalized_samples
from pozify.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class PushUpRepCounter(ExerciseRepCounter):
    exercise = "push_up"

    def build_signal(self, sequence: PoseSequence) -> tuple[list[SignalSample], dict[str, Any]]:
        hip_y = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in sequence.frames]
        shoulder_y = [average_axis(frame, ("left_shoulder", "right_shoulder"), "y") for frame in sequence.frames]
        elbow_bend = [
            mean_optional(
                [
                    None if angle is None else max(0.0, 180.0 - angle)
                    for angle in (
                        angle_deg(frame, "left_shoulder", "left_elbow", "left_wrist"),
                        angle_deg(frame, "right_shoulder", "right_elbow", "right_wrist"),
                    )
                ]
            )
            for frame in sequence.frames
        ]
        body_line = [body_line_score(frame) for frame in sequence.frames]
        chest_proxy = [
            mean_optional([shoulder_value, hip_value])
            for shoulder_value, hip_value in zip(shoulder_y, hip_y, strict=False)
        ]
        samples, signal_range = normalized_samples(sequence, combine(chest_proxy, elbow_bend, weight=0.25))
        return samples, {
            "selected_signal": "chest_y_plus_elbow_bend",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

