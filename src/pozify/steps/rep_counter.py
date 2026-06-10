from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pozify.contracts import ExerciseClassification, PoseSequence, Rep, Reps
from pozify.steps.rep_signals import (
    SignalSample,
    angle_deg,
    average_axis,
    body_line_score,
    normalize_optional,
    samples_from_values,
    smooth_signal,
)
from pozify.steps.rep_state_machine import RepSegment, find_local_extrema, segment_low_high_low


MIN_CYCLE_FRAMES = 12
MIN_PHASE_FRAMES = 4
MIN_USABLE_SIGNAL_SAMPLES = 9
MIN_SIGNAL_RANGE = 0.22


def _mean_optional(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _combine(primary: list[float | None], secondary: list[float | None], *, weight: float) -> list[float | None]:
    normalized_secondary = normalize_optional(secondary)
    combined: list[float | None] = []
    for primary_value, secondary_value in zip(primary, normalized_secondary, strict=False):
        if primary_value is None:
            combined.append(None)
            continue
        if secondary_value is None:
            combined.append(primary_value)
            continue
        combined.append(primary_value + secondary_value * weight)
    return combined


def _primary_signal_for_exercise(sequence: PoseSequence, exercise: str) -> tuple[list[SignalSample], dict[str, Any]]:
    hip_y = [average_axis(frame, ("left_hip", "right_hip"), "y") for frame in sequence.frames]
    shoulder_y = [average_axis(frame, ("left_shoulder", "right_shoulder"), "y") for frame in sequence.frames]
    wrist_y = [average_axis(frame, ("left_wrist", "right_wrist"), "y") for frame in sequence.frames]
    knee_bend = [
        _mean_optional(
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
    elbow_bend = [
        _mean_optional(
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

    if exercise == "squat":
        raw_signal = _combine(hip_y, knee_bend, weight=0.35)
        selected_signal = "hip_y_plus_knee_bend"
    elif exercise == "shoulder_press":
        inverted_wrist = [None if value is None else -value for value in wrist_y]
        raw_signal = _combine(inverted_wrist, [None if value is None else -value for value in elbow_bend], weight=0.2)
        selected_signal = "negative_wrist_y_plus_elbow_extension_proxy"
    else:
        chest_proxy = [
            _mean_optional([shoulder_value, hip_value])
            for shoulder_value, hip_value in zip(shoulder_y, hip_y, strict=False)
        ]
        raw_signal = _combine(chest_proxy, elbow_bend, weight=0.25)
        selected_signal = "chest_y_plus_elbow_bend"

    smoothed_signal = smooth_signal(raw_signal)
    normalized_signal = normalize_optional(smoothed_signal)
    samples = samples_from_values(sequence, normalized_signal)
    return samples, {
        "selected_signal": selected_signal,
        "raw_signal_range": (
            round(max((value for value in normalized_signal if value is not None), default=0.0)
            - min((value for value in normalized_signal if value is not None), default=0.0), 4)
        ),
        "usable_signal_samples": len(samples),
        "body_line_mean": round(_mean_optional(body_line) or 0.0, 4),
    }


def _segments_to_reps(segments: list[RepSegment]) -> list[Rep]:
    return [
        Rep(
            rep_id=index + 1,
            start_frame=segment.start.frame_index,
            mid_frame=segment.middle.frame_index,
            end_frame=segment.end.frame_index,
            start_sec=round(segment.start.timestamp_sec, 3),
            mid_sec=round(segment.middle.timestamp_sec, 3),
            end_sec=round(segment.end.timestamp_sec, 3),
        )
        for index, segment in enumerate(segments)
    ]


def _partial_reps(
    sequence: PoseSequence,
    segments: list[RepSegment],
    samples: list[SignalSample],
    *,
    signal_range: float,
) -> list[dict[str, Any]]:
    if not samples:
        return [{"reason": "low_signal_quality"}]

    partials: list[dict[str, Any]] = []
    if not segments:
        if signal_range >= MIN_SIGNAL_RANGE * 0.7:
            partials.append(
                {
                    "reason": "insufficient_rom",
                    "start_frame": samples[0].frame_index,
                    "end_frame": samples[-1].frame_index,
                    "start_sec": round(samples[0].timestamp_sec, 3),
                    "end_sec": round(samples[-1].timestamp_sec, 3),
                }
            )
        return partials

    first_segment = segments[0]
    if first_segment.start.frame_index - samples[0].frame_index >= MIN_PHASE_FRAMES:
        partials.append(
            {
                "reason": "starts_mid_rep",
                "start_frame": samples[0].frame_index,
                "end_frame": first_segment.start.frame_index,
                "start_sec": round(samples[0].timestamp_sec, 3),
                "end_sec": round(first_segment.start.timestamp_sec, 3),
            }
        )

    last_segment = segments[-1]
    if samples[-1].frame_index - last_segment.end.frame_index >= MIN_PHASE_FRAMES:
        partials.append(
            {
                "reason": "ends_mid_rep",
                "start_frame": last_segment.end.frame_index,
                "end_frame": samples[-1].frame_index,
                "start_sec": round(last_segment.end.timestamp_sec, 3),
                "end_sec": round(samples[-1].timestamp_sec, 3),
            }
        )
    return partials


def run(classification: ExerciseClassification, sequence: PoseSequence) -> tuple[Reps, dict[str, Any]]:
    if classification.exercise == "unknown":
        reps = Reps(exercise=classification.exercise, reps=[], partial_reps=[{"reason": "unknown_exercise"}])
        return reps, {"selected_signal": "none", "thresholds": {}, "extrema": [], "accepted_reps": []}

    samples, debug = _primary_signal_for_exercise(sequence, classification.exercise)
    signal_range = debug["raw_signal_range"]
    extrema = find_local_extrema(samples)
    segments = (
        segment_low_high_low(
            extrema,
            min_cycle_frames=MIN_CYCLE_FRAMES,
            min_phase_frames=MIN_PHASE_FRAMES,
            min_amplitude=max(MIN_SIGNAL_RANGE, signal_range * 0.35),
        )
        if len(samples) >= MIN_USABLE_SIGNAL_SAMPLES
        else []
    )
    partial_reps = _partial_reps(sequence, segments, samples, signal_range=signal_range)
    if sequence.pose_valid_ratio < 0.8:
        partial_reps.append({"reason": "low_pose_valid_ratio"})

    reps = Reps(
        exercise=classification.exercise,
        reps=_segments_to_reps(segments),
        partial_reps=partial_reps,
    )
    debug_payload = {
        **debug,
        "thresholds": {
            "min_cycle_frames": MIN_CYCLE_FRAMES,
            "min_phase_frames": MIN_PHASE_FRAMES,
            "min_amplitude": round(max(MIN_SIGNAL_RANGE, signal_range * 0.35), 4),
        },
        "extrema": [
            {
                "kind": extrema_item.kind,
                "frame_index": extrema_item.sample.frame_index,
                "timestamp_sec": round(extrema_item.sample.timestamp_sec, 3),
                "value": round(extrema_item.sample.value, 4),
            }
            for extrema_item in extrema
        ],
        "accepted_reps": [
            {
                "start": asdict(segment.start),
                "middle": asdict(segment.middle),
                "end": asdict(segment.end),
                "amplitude": round(segment.amplitude, 4),
            }
            for segment in segments
        ],
    }
    return reps, debug_payload
