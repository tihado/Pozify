from __future__ import annotations

from dataclasses import dataclass
import math

from pozify.contracts import PoseFrame, PoseSequence


@dataclass(frozen=True)
class SignalSample:
    frame_index: int
    timestamp_sec: float
    value: float


def _landmark_values(frame: PoseFrame, name: str) -> dict[str, float] | None:
    return frame.landmarks.get(name)


def landmark_axis(frame: PoseFrame, name: str, axis: str) -> float | None:
    values = _landmark_values(frame, name)
    if values is None:
        return None
    return values.get(f"smoothed_{axis}", values.get(axis))


def average_axis(frame: PoseFrame, names: tuple[str, ...], axis: str) -> float | None:
    values = [landmark_axis(frame, name, axis) for name in names]
    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return None
    return sum(usable_values) / len(usable_values)


def angle_deg(frame: PoseFrame, first: str, middle: str, last: str) -> float | None:
    ax = landmark_axis(frame, first, "x")
    ay = landmark_axis(frame, first, "y")
    bx = landmark_axis(frame, middle, "x")
    by = landmark_axis(frame, middle, "y")
    cx = landmark_axis(frame, last, "x")
    cy = landmark_axis(frame, last, "y")
    if None in {ax, ay, bx, by, cx, cy}:
        return None

    abx = ax - bx
    aby = ay - by
    cbx = cx - bx
    cby = cy - by
    denom = math.hypot(abx, aby) * math.hypot(cbx, cby)
    if denom <= 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (abx * cbx + aby * cby) / denom))
    return math.degrees(math.acos(cosine))


def body_line_score(frame: PoseFrame) -> float | None:
    shoulder_y = average_axis(frame, ("left_shoulder", "right_shoulder"), "y")
    hip_y = average_axis(frame, ("left_hip", "right_hip"), "y")
    ankle_y = average_axis(frame, ("left_ankle", "right_ankle"), "y")
    if None in {shoulder_y, hip_y, ankle_y}:
        return None
    return 1.0 - abs((shoulder_y + ankle_y) / 2.0 - hip_y)


def smooth_signal(values: list[float | None], window_radius: int = 2) -> list[float | None]:
    smoothed: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            smoothed.append(None)
            continue
        window = values[max(0, index - window_radius) : index + window_radius + 1]
        usable_window = [item for item in window if item is not None]
        smoothed.append(sum(usable_window) / len(usable_window) if usable_window else None)
    return smoothed


def normalize_optional(values: list[float | None]) -> list[float | None]:
    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return [None for _ in values]
    min_value = min(usable_values)
    max_value = max(usable_values)
    value_range = max_value - min_value
    if value_range <= 1e-6:
        return [0.0 if value is not None else None for value in values]
    return [
        None if value is None else (value - min_value) / value_range
        for value in values
    ]


def samples_from_values(sequence: PoseSequence, values: list[float | None]) -> list[SignalSample]:
    return [
        SignalSample(
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            value=value,
        )
        for frame, value in zip(sequence.frames, values, strict=False)
        if value is not None
    ]
