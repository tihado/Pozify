from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence
from pozify.steps.rep_counters.base import ExerciseRepCounter, combine, mean_optional, normalized_samples
from pozify.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class SquatRepCounter(ExerciseRepCounter):
    exercise = "squat"

    def build_signal(self, sequence: PoseSequence) -> tuple[list[SignalSample], dict[str, Any]]:
        hip_y = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in sequence.frames]
        knee_bend = [
            mean_optional(
                [
                    None if angle is None else max(0.0, 180.0 - angle)
                    for angle in (
                        angle_deg(frame, "left_hip", "left_knee", "left_ankle"),
                        angle_deg(frame, "right_hip", "right_knee", "right_ankle"),
                    )
                ]
            )
            for frame in sequence.frames
        ]
        body_line = [body_line_score(frame) for frame in sequence.frames]
        samples, signal_range = normalized_samples(sequence, combine(hip_y, knee_bend, weight=0.35))
        return samples, {
            "selected_signal": "hip_y_plus_knee_bend",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

