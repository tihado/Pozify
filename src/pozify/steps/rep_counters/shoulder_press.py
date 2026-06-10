from __future__ import annotations

from typing import Any

from pozify.contracts import PoseSequence
from pozify.steps.rep_counters.base import ExerciseRepCounter, combine, mean_optional, normalized_samples
from pozify.steps.rep_signals import SignalSample, angle_deg, average_axis, body_line_score


class ShoulderPressRepCounter(ExerciseRepCounter):
    exercise = "shoulder_press"

    def build_signal(self, sequence: PoseSequence) -> tuple[list[SignalSample], dict[str, Any]]:
        wrist_y = [average_axis(frame, ("left_wrist", "right_wrist"), "y") for frame in sequence.frames]
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
        inverted_wrist = [None if value is None else -value for value in wrist_y]
        inverted_elbow_bend = [None if value is None else -value for value in elbow_bend]
        raw_signal = combine(inverted_wrist, inverted_elbow_bend, weight=0.2)
        samples, signal_range = normalized_samples(sequence, raw_signal)
        return samples, {
            "selected_signal": "negative_wrist_y_plus_elbow_extension_proxy",
            "raw_signal_range": signal_range,
            "usable_signal_samples": len(samples),
            "body_line_mean": round(mean_optional(body_line) or 0.0, 4),
        }

