from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Callable

from pozify.contracts import (
    PoseFrame,
    PoseSequence,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)
from pozify.exercises import ExerciseStrategy
from pozify.exercises.shared.analyzer import (
    mean_optional,
    round_optional,
    safe_ratio,
    score,
    usable,
    value_series,
)
from pozify.steps.rep_signals import average_axis


NumberGetter = Callable[[PoseFrame], float | None]


def _std(values: list[float | None]) -> float | None:
    usable_values = usable(values)
    if len(usable_values) < 2:
        return 0.0 if usable_values else None
    return pstdev(usable_values)


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
    return score(mean_optional(values) if values else 0.0)


def _smoothness_score(signal_values: list[float | None]) -> tuple[float, float | None]:
    usable_values = usable(signal_values)
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
    return score(1.0 - jerk * 8.0), jerk


def _pause_duration(
    frames: list[PoseFrame],
    signal_values: list[float | None],
    *,
    target: str,
) -> float:
    usable_values = usable(signal_values)
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
    stability_axis = value_series(
        frames,
        lambda frame: average_axis(frame, ("left_hip", "right_hip"), "x"),
    )
    stability_noise = _std(stability_axis) or 0.0

    return {
        "rep_duration_sec": duration,
        "eccentric_duration_sec": eccentric_duration,
        "concentric_duration_sec": concentric_duration,
        "tempo_ratio": round_optional(safe_ratio(eccentric_duration, concentric_duration), 2),
        "top_pause_sec": _pause_duration(frames, primary_signal, target="top"),
        "bottom_pause_sec": _pause_duration(frames, primary_signal, target="bottom"),
        "smoothness_score": smoothness_score,
        "jerk_score": round_optional(jerk_score, 4),
        "landmark_confidence": _mean_visibility(frames),
        "hip_lateral_drift": round_optional(stability_noise, 4),
    }


def _primary_signal(exercise_key: str, frames: list[PoseFrame]) -> list[float | None]:
    if exercise_key == "shoulder_press":
        return value_series(
            frames,
            lambda frame: average_axis(frame, ("left_wrist", "right_wrist"), "y"),
        )
    if exercise_key == "push_up":
        return value_series(
            frames,
            lambda frame: mean_optional(
                [
                    average_axis(frame, ("left_shoulder", "right_shoulder"), "y"),
                    average_axis(frame, ("left_hip", "right_hip"), "y"),
                ]
            ),
        )
    return value_series(frames, lambda frame: average_axis(frame, ("left_hip", "right_hip"), "y"))


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
    exercise: ExerciseStrategy,
    reps: Reps,
    sequence: PoseSequence,
) -> RepAnalysis:
    draft_items: list[tuple[Rep, dict[str, Any], float, float, float, list[str]]] = []
    for rep in reps.reps:
        frames = _frames_for_rep(sequence, rep)
        primary_signal = _primary_signal(exercise.exercise, frames)
        common_metrics = _common_metrics(rep, frames, primary_signal)
        exercise_metrics, rom_score, stability_score, symmetry_score, hints = exercise.metrics(frames)
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
            score(1.0 - abs(duration - average_duration) / max(average_duration, 0.1))
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
        exercise=exercise.exercise,
        items=items,
        aggregate_metrics=aggregate_metrics,
    )
