from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Callable

from pozify.contracts import (
    ExerciseClassification,
    PoseFrame,
    PoseSequence,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)
from pozify.steps.rep_signals import angle_deg, average_axis, body_line_score, landmark_axis


NumberGetter = Callable[[PoseFrame], float | None]
ExerciseMetricResult = tuple[dict[str, Any], float, float, float, list[str]]


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _score(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 2)


def _usable(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _mean(values: list[float | None]) -> float | None:
    usable_values = _usable(values)
    if not usable_values:
        return None
    return sum(usable_values) / len(usable_values)


def _min(values: list[float | None]) -> float | None:
    usable_values = _usable(values)
    return min(usable_values) if usable_values else None


def _max(values: list[float | None]) -> float | None:
    usable_values = _usable(values)
    return max(usable_values) if usable_values else None


def _range(values: list[float | None]) -> float | None:
    usable_values = _usable(values)
    if not usable_values:
        return None
    return max(usable_values) - min(usable_values)


def _std(values: list[float | None]) -> float | None:
    usable_values = _usable(values)
    if len(usable_values) < 2:
        return 0.0 if usable_values else None
    return pstdev(usable_values)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-6:
        return None
    return numerator / denominator


def _width(frame: PoseFrame, left: str, right: str) -> float | None:
    left_x = landmark_axis(frame, left, "x")
    right_x = landmark_axis(frame, right, "x")
    if left_x is None or right_x is None:
        return None
    return abs(right_x - left_x)


def _mean_pair(
    frame: PoseFrame,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
) -> float | None:
    values = [angle_deg(frame, *first), angle_deg(frame, *second)]
    return _mean(values)


def _side_delta(
    frame: PoseFrame,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
) -> float | None:
    first_value = angle_deg(frame, *first)
    second_value = angle_deg(frame, *second)
    if first_value is None or second_value is None:
        return None
    return abs(first_value - second_value)


def _torso_lean_deg(frame: PoseFrame, side: str) -> float | None:
    shoulder_x = landmark_axis(frame, f"{side}_shoulder", "x")
    shoulder_y = landmark_axis(frame, f"{side}_shoulder", "y")
    hip_x = landmark_axis(frame, f"{side}_hip", "x")
    hip_y = landmark_axis(frame, f"{side}_hip", "y")
    if None in {shoulder_x, shoulder_y, hip_x, hip_y}:
        return None
    horizontal_offset = abs(float(shoulder_x) - float(hip_x))
    vertical_offset = abs(float(shoulder_y) - float(hip_y))
    if vertical_offset <= 1e-6:
        return None
    from math import atan2, degrees

    return degrees(atan2(horizontal_offset, vertical_offset))


def _frames_for_rep(sequence: PoseSequence, rep: Rep) -> list[PoseFrame]:
    frames = [
        frame
        for frame in sequence.frames
        if rep.start_frame <= frame.frame_index <= rep.end_frame
    ]
    if frames:
        return frames

    if not sequence.frames:
        return []
    closest = min(
        sequence.frames,
        key=lambda frame: min(
            abs(frame.frame_index - rep.start_frame),
            abs(frame.frame_index - rep.mid_frame),
            abs(frame.frame_index - rep.end_frame),
        ),
    )
    return [closest]


def _value_series(frames: list[PoseFrame], getter: NumberGetter) -> list[float | None]:
    return [getter(frame) for frame in frames]


def _mean_visibility(frames: list[PoseFrame]) -> float:
    values: list[float | None] = []
    for frame in frames:
        if "mean_visibility" in frame.pose_quality:
            values.append(float(frame.pose_quality["mean_visibility"]))
            continue
        landmark_values = [
            landmark.get("visibility")
            for landmark in frame.landmarks.values()
            if landmark.get("visibility") is not None
        ]
        values.extend(float(value) for value in landmark_values)
    return _score(_mean(values) if values else 0.0)


def _smoothness_score(signal_values: list[float | None]) -> tuple[float, float | None]:
    usable_values = _usable(signal_values)
    if len(usable_values) < 4:
        return 0.5, None

    deltas = [
        usable_values[index] - usable_values[index - 1]
        for index in range(1, len(usable_values))
    ]
    jerks = [deltas[index] - deltas[index - 1] for index in range(1, len(deltas))]
    if not jerks:
        return 0.5, None
    jerk = mean(abs(value) for value in jerks)
    return _score(1.0 - jerk * 8.0), jerk


def _pause_duration(
    frames: list[PoseFrame],
    signal_values: list[float | None],
    *,
    target: str,
) -> float:
    usable_values = _usable(signal_values)
    if len(usable_values) < 3 or len(frames) < 3:
        return 0.0

    min_value = min(usable_values)
    max_value = max(usable_values)
    tolerance = max((max_value - min_value) * 0.08, 0.01)
    if target == "bottom":
        active = [value is not None and value >= max_value - tolerance for value in signal_values]
    else:
        active = [value is not None and value <= min_value + tolerance for value in signal_values]

    longest = 0
    current = 0
    for item in active:
        if item:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    if longest <= 1:
        return 0.0
    frame_duration = (frames[-1].timestamp_sec - frames[0].timestamp_sec) / max(1, len(frames) - 1)
    return round(longest * frame_duration, 2)


def _common_metrics(
    rep: Rep,
    frames: list[PoseFrame],
    primary_signal: list[float | None],
) -> dict[str, Any]:
    eccentric_duration = round(max(0.0, rep.mid_sec - rep.start_sec), 2)
    concentric_duration = round(max(0.0, rep.end_sec - rep.mid_sec), 2)
    duration = round(max(0.0, rep.end_sec - rep.start_sec), 2)
    smoothness_score, jerk_score = _smoothness_score(primary_signal)
    stability_axis = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_hip", "right_hip"), "x"),
    )
    stability_noise = _std(stability_axis) or 0.0

    return {
        "rep_duration_sec": duration,
        "eccentric_duration_sec": eccentric_duration,
        "concentric_duration_sec": concentric_duration,
        "tempo_ratio": _round(_safe_ratio(eccentric_duration, concentric_duration), 2),
        "top_pause_sec": _pause_duration(frames, primary_signal, target="top"),
        "bottom_pause_sec": _pause_duration(frames, primary_signal, target="bottom"),
        "smoothness_score": smoothness_score,
        "jerk_score": _round(jerk_score, 4),
        "landmark_confidence": _mean_visibility(frames),
        "hip_lateral_drift": _round(stability_noise, 4),
    }


def _squat_metrics(frames: list[PoseFrame]) -> ExerciseMetricResult:
    knee_angles = _value_series(
        frames,
        lambda frame: _mean_pair(
            frame,
            ("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle"),
        ),
    )
    hip_angles = _value_series(
        frames,
        lambda frame: _mean_pair(
            frame,
            ("left_shoulder", "left_hip", "left_knee"),
            ("right_shoulder", "right_hip", "right_knee"),
        ),
    )
    knee_deltas = _value_series(
        frames,
        lambda frame: _side_delta(
            frame,
            ("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle"),
        ),
    )
    hip_deltas = _value_series(
        frames,
        lambda frame: _side_delta(
            frame,
            ("left_shoulder", "left_hip", "left_knee"),
            ("right_shoulder", "right_hip", "right_knee"),
        ),
    )
    hip_y = _value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))
    knee_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_knee", "right_knee"), "y"),
    )
    ankle_width = _value_series(frames, lambda frame: _width(frame, "left_ankle", "right_ankle"))
    shoulder_width = _value_series(
        frames,
        lambda frame: _width(frame, "left_shoulder", "right_shoulder"),
    )
    knee_width = _value_series(frames, lambda frame: _width(frame, "left_knee", "right_knee"))
    torso_lean = _value_series(
        frames,
        lambda frame: _mean(
            [_torso_lean_deg(frame, "left"), _torso_lean_deg(frame, "right")]
        ),
    )

    min_knee = _min(knee_angles)
    max_knee = _max(knee_angles)
    min_hip = _min(hip_angles)
    max_hip = _max(hip_angles)
    hip_depth_delta = None
    max_hip_y = _max(hip_y)
    mean_knee_y = _mean(knee_y)
    if max_hip_y is not None and mean_knee_y is not None:
        hip_depth_delta = max_hip_y - mean_knee_y

    stance_ratio = _safe_ratio(_mean(ankle_width), _mean(shoulder_width))
    knee_tracking_ratio = _safe_ratio(_mean(knee_width), _mean(ankle_width))
    valgus_proxy = None if knee_tracking_ratio is None else max(0.0, 1.0 - knee_tracking_ratio)
    symmetry_delta = _mean(knee_deltas + hip_deltas) or 0.0
    stability_noise = (_std(hip_y) or 0.0) + (_std(knee_width) or 0.0)

    knee_rom = 0.0 if min_knee is None or max_knee is None else max_knee - min_knee
    depth_score = _score((hip_depth_delta + 0.08) / 0.18) if hip_depth_delta is not None else 0.5
    angle_score = _score(knee_rom / 65.0)
    rom_score = _score(angle_score * 0.55 + depth_score * 0.45)
    stability_score = _score(1.0 - stability_noise * 5.0)
    symmetry_score = _score(1.0 - symmetry_delta / 45.0)

    hip_x = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_hip", "right_hip"), "x"),
    )
    metrics = {
        "min_knee_angle_deg": _round(min_knee),
        "max_knee_angle_deg": _round(max_knee),
        "min_hip_angle_deg": _round(min_hip),
        "max_hip_angle_deg": _round(max_hip),
        "hip_depth_delta": _round(hip_depth_delta, 4),
        "hip_depth_relative_to_knee": (
            "below_parallel"
            if hip_depth_delta is not None and hip_depth_delta >= 0.03
            else "parallel"
            if hip_depth_delta is not None and hip_depth_delta >= -0.03
            else "above_parallel"
        ),
        "max_torso_lean_deg": _round(_max(torso_lean)),
        "knee_valgus_proxy": _round(valgus_proxy, 4),
        "knee_tracking_score": _score(1.0 - (valgus_proxy or 0.0)),
        "stance_width_ratio": _round(stance_ratio, 3),
        "hip_shift": _round(_std(hip_x), 4),
        "bottom_stability_score": stability_score,
    }
    hints = []
    if stance_ratio is not None and stance_ratio > 1.35:
        hints.append("wide_squat_stance")
    elif stance_ratio is not None and stance_ratio < 0.85:
        hints.append("narrow_squat_stance")
    return metrics, rom_score, stability_score, symmetry_score, hints


def _push_up_metrics(frames: list[PoseFrame]) -> ExerciseMetricResult:
    elbow_angles = _value_series(
        frames,
        lambda frame: _mean_pair(
            frame,
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        ),
    )
    elbow_deltas = _value_series(
        frames,
        lambda frame: _side_delta(
            frame,
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        ),
    )
    body_line = _value_series(frames, body_line_score)
    shoulder_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
    )
    hip_y = _value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))
    ankle_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_ankle", "right_ankle"), "y"),
    )
    hand_width = _value_series(frames, lambda frame: _width(frame, "left_wrist", "right_wrist"))
    shoulder_width = _value_series(
        frames,
        lambda frame: _width(frame, "left_shoulder", "right_shoulder"),
    )
    elbow_width = _value_series(frames, lambda frame: _width(frame, "left_elbow", "right_elbow"))
    knee_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_knee", "right_knee"), "y"),
    )

    min_elbow = _min(elbow_angles)
    max_elbow = _max(elbow_angles)
    elbow_rom = 0.0 if min_elbow is None or max_elbow is None else max_elbow - min_elbow
    chest_depth = _range(shoulder_y) or 0.0
    hand_width_ratio = _safe_ratio(_mean(hand_width), _mean(shoulder_width))
    elbow_flare = _safe_ratio(_mean(elbow_width), _mean(shoulder_width))
    body_line_mean = _mean(body_line)
    hip_sag_score = None
    if body_line_mean is not None:
        hip_sag_score = max(0.0, 1.0 - body_line_mean)
    knee_support_score = 0.0
    mean_knee_y = _mean(knee_y)
    mean_ankle_y = _mean(ankle_y)
    if mean_knee_y is not None and mean_ankle_y is not None:
        knee_support_score = _score(1.0 - abs(mean_ankle_y - mean_knee_y) / 0.18)

    rom_score = _score((elbow_rom / 80.0) * 0.65 + (chest_depth / 0.16) * 0.35)
    hip_stability = 1.0 - min(1.0, (_std(hip_y) or 0.0) * 4.0)
    stability_score = _score(((body_line_mean or 0.5) * 0.75) + hip_stability * 0.25)
    symmetry_score = _score(1.0 - ((_mean(elbow_deltas) or 0.0) / 45.0))

    metrics = {
        "min_elbow_angle_deg": _round(min_elbow),
        "max_elbow_angle_deg": _round(max_elbow),
        "body_line_score": _round(body_line_mean),
        "hip_sag_score": _round(hip_sag_score),
        "hip_pike_score": _round(max(0.0, ((body_line_mean or 1.0) - 1.0) * -1.0), 4),
        "chest_depth_proxy": _round(chest_depth, 4),
        "hand_width_ratio": _round(hand_width_ratio, 3),
        "elbow_flare_ratio": _round(elbow_flare, 3),
        "lockout_quality": _score(((max_elbow or 120.0) - 120.0) / 55.0),
        "knee_support_score": knee_support_score,
    }
    hints = []
    if hand_width_ratio is not None and hand_width_ratio > 1.45:
        hints.append("wide_grip_push_up")
    elif hand_width_ratio is not None and hand_width_ratio < 0.95:
        hints.append("close_grip_push_up")
    if knee_support_score >= 0.65:
        hints.append("knee_push_up")
    return metrics, rom_score, stability_score, symmetry_score, hints


def _shoulder_press_metrics(frames: list[PoseFrame]) -> ExerciseMetricResult:
    elbow_angles = _value_series(
        frames,
        lambda frame: _mean_pair(
            frame,
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        ),
    )
    elbow_deltas = _value_series(
        frames,
        lambda frame: _side_delta(
            frame,
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
        ),
    )
    wrist_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "y"),
    )
    wrist_x = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "x"),
    )
    left_wrist_y = _value_series(frames, lambda frame: landmark_axis(frame, "left_wrist", "y"))
    right_wrist_y = _value_series(frames, lambda frame: landmark_axis(frame, "right_wrist", "y"))
    wrist_asymmetry = [
        abs(left - right)
        for left, right in zip(left_wrist_y, right_wrist_y, strict=False)
        if left is not None and right is not None
    ]
    shoulder_y = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
    )
    hip_y = _value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))

    min_elbow = _min(elbow_angles)
    max_elbow = _max(elbow_angles)
    wrist_travel = _range(wrist_y) or 0.0
    wrist_lateral_drift = _range(wrist_x) or 0.0
    lockout_quality = _score(((max_elbow or 120.0) - 120.0) / 55.0)
    verticality = _score(1.0 - wrist_lateral_drift / max(0.01, wrist_travel))
    asymmetry = mean(wrist_asymmetry) if wrist_asymmetry else 0.0
    back_arch_proxy = abs((_range(hip_y) or 0.0) - (_range(shoulder_y) or 0.0))

    rom_score = _score((wrist_travel / 0.28) * 0.55 + lockout_quality * 0.45)
    stability_score = _score(verticality * 0.65 + (1.0 - min(1.0, back_arch_proxy * 4.0)) * 0.35)
    symmetry_score = _score(1.0 - max(asymmetry * 5.0, (_mean(elbow_deltas) or 0.0) / 45.0))

    metrics = {
        "min_elbow_angle_deg": _round(min_elbow),
        "max_elbow_angle_deg": _round(max_elbow),
        "wrist_path_verticality": verticality,
        "lockout_quality": lockout_quality,
        "wrist_height_asymmetry": _round(asymmetry, 4),
        "left_right_wrist_delta": _round(asymmetry, 4),
        "back_arch_proxy": _round(back_arch_proxy, 4),
        "overhead_stability_score": stability_score,
        "wrist_travel": _round(wrist_travel, 4),
    }
    hints = []
    if lockout_quality < 0.65:
        hints.append("partial_press")
    if asymmetry > 0.12:
        hints.append("asymmetric_press")
    return metrics, rom_score, stability_score, symmetry_score, hints


def _unknown_metrics(frames: list[PoseFrame]) -> ExerciseMetricResult:
    movement_signal = _value_series(
        frames,
        lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"),
    )
    movement = _range(movement_signal) or 0.0
    confidence = _mean_visibility(frames)
    metrics = {
        "movement_consistency_score": _score(1.0 - movement * 4.0),
        "landmark_confidence": confidence,
    }
    return metrics, 0.0, metrics["movement_consistency_score"], confidence, []


def _exercise_metrics(
    exercise: str,
    frames: list[PoseFrame],
) -> ExerciseMetricResult:
    if exercise == "squat":
        return _squat_metrics(frames)
    if exercise == "push_up":
        return _push_up_metrics(frames)
    if exercise == "shoulder_press":
        return _shoulder_press_metrics(frames)
    return _unknown_metrics(frames)


def _primary_signal(exercise: str, frames: list[PoseFrame]) -> list[float | None]:
    if exercise == "shoulder_press":
        return _value_series(
            frames,
            lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "y"),
        )
    if exercise == "push_up":
        return _value_series(
            frames,
            lambda frame: _mean(
                [
                    average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
                    average_axis(frame, ("left_hip", "right_hip"), "y"),
                ]
            ),
        )
    return _value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))


def _aggregate_numeric(items: list[RepAnalysisItem], metric_name: str) -> float | None:
    values = [
        item.metrics.get(metric_name)
        for item in items
        if isinstance(item.metrics.get(metric_name), (int, float))
    ]
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 4)


def _fatigue_trend(items: list[RepAnalysisItem]) -> float:
    if len(items) < 2:
        return 0.0
    first = items[0].range_of_motion_score
    last = items[-1].range_of_motion_score
    return round(last - first, 4)


def run(
    classification: ExerciseClassification,
    reps: Reps,
    sequence: PoseSequence,
) -> RepAnalysis:
    draft_items: list[tuple[Rep, dict[str, Any], float, float, float, list[str]]] = []
    for rep in reps.reps:
        frames = _frames_for_rep(sequence, rep)
        primary_signal = _primary_signal(classification.exercise, frames)
        common_metrics = _common_metrics(rep, frames, primary_signal)
        exercise_metrics, rom_score, stability_score, symmetry_score, hints = _exercise_metrics(
            classification.exercise,
            frames,
        )
        metrics = {**common_metrics, **exercise_metrics}
        draft_items.append((rep, metrics, rom_score, stability_score, symmetry_score, hints))

    average_duration = (
        mean(item[1]["rep_duration_sec"] for item in draft_items)
        if draft_items
        else 0.0
    )
    items: list[RepAnalysisItem] = []
    for rep, metrics, rom_score, stability_score, symmetry_score, hints in draft_items:
        duration = metrics["rep_duration_sec"]
        metrics["tempo_consistency_score"] = (
            _score(1.0 - abs(duration - average_duration) / max(average_duration, 0.1))
            if average_duration
            else 0.0
        )
        items.append(
            RepAnalysisItem(
                rep_id=rep.rep_id,
                duration_sec=duration,
                range_of_motion_score=rom_score,
                stability_score=stability_score,
                symmetry_score=symmetry_score,
                metrics=metrics,
                variation_hints=sorted(set(hints)),
            )
        )

    aggregate_metrics = {
        "avg_rom_score": (
            round(mean(item.range_of_motion_score for item in items), 2) if items else 0.0
        ),
        "avg_stability_score": (
            round(mean(item.stability_score for item in items), 2) if items else 0.0
        ),
        "avg_symmetry_score": (
            round(mean(item.symmetry_score for item in items), 2) if items else 0.0
        ),
        "avg_rep_duration_sec": (
            round(mean(item.duration_sec for item in items), 2) if items else 0.0
        ),
        "avg_tempo_consistency_score": _aggregate_numeric(items, "tempo_consistency_score") or 0.0,
        "avg_landmark_confidence": (
            _aggregate_numeric(items, "landmark_confidence") or sequence.pose_valid_ratio
        ),
        "fatigue_trend_rom_delta": _fatigue_trend(items),
        "pose_valid_ratio": sequence.pose_valid_ratio,
    }

    for metric_name in (
        "hand_width_ratio",
        "stance_width_ratio",
        "bottom_pause_sec",
        "lockout_quality",
        "wrist_height_asymmetry",
        "wrist_travel",
        "knee_support_score",
    ):
        aggregate_value = _aggregate_numeric(items, metric_name)
        if aggregate_value is not None:
            aggregate_metrics[f"avg_{metric_name}"] = aggregate_value

    return RepAnalysis(
        exercise=classification.exercise,
        items=items,
        aggregate_metrics=aggregate_metrics,
    )
