from __future__ import annotations

from pozify.contracts import ExerciseClassification, RepAnalysis, UserProfile, Variation
from pozify.exercise_catalog import get_exercise_spec


def _metric(analysis: RepAnalysis, name: str) -> float | None:
    value = analysis.aggregate_metrics.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _confidence(base: float, analysis: RepAnalysis, supporting_metric: float | None) -> float:
    rep_bonus = min(0.12, len(analysis.items) * 0.03)
    metric_bonus = 0.0 if supporting_metric is None else min(0.1, abs(supporting_metric) * 0.03)
    pose_bonus = min(0.08, float(analysis.aggregate_metrics.get("pose_valid_ratio", 0.0)) * 0.08)
    return round(min(0.95, base + rep_bonus + metric_bonus + pose_bonus), 2)


def _push_up_variation(analysis: RepAnalysis) -> tuple[str, float, list[str]]:
    hand_width = _metric(analysis, "avg_hand_width_ratio")
    knee_support = _metric(analysis, "avg_knee_support_score")
    not_issues: list[str] = []

    if knee_support is not None and knee_support >= 0.8:
        return "knee_push_up", _confidence(0.74, analysis, knee_support), ["knee_contact"]
    if hand_width is not None and hand_width >= 1.45:
        return "wide_grip_push_up", _confidence(0.72, analysis, hand_width), ["wide_hand_placement"]
    if hand_width is not None and hand_width <= 0.95:
        return (
            "close_grip_push_up",
            _confidence(0.72, analysis, 1.0 - hand_width),
            ["close_hand_placement"],
        )

    if hand_width is None:
        not_issues.append("hand_width_unverified")
    return "standard_push_up", _confidence(0.62, analysis, hand_width), not_issues


def _squat_variation(analysis: RepAnalysis) -> tuple[str, float, list[str]]:
    stance_width = _metric(analysis, "avg_stance_width_ratio")
    bottom_pause = _metric(analysis, "avg_bottom_pause_sec")

    if bottom_pause is not None and bottom_pause >= 0.4:
        return "pause_squat", _confidence(0.76, analysis, bottom_pause), ["bottom_pause"]
    if stance_width is not None and stance_width >= 1.35:
        return "wide_squat_stance", _confidence(0.72, analysis, stance_width), ["wide_stance"]
    if stance_width is not None and stance_width <= 0.85:
        return (
            "narrow_squat_stance",
            _confidence(0.72, analysis, 1.0 - stance_width),
            ["narrow_stance"],
        )

    not_issues = ["stance_width_unverified"] if stance_width is None else []
    return "normal_squat_stance", _confidence(0.62, analysis, stance_width), not_issues


def _shoulder_press_variation(analysis: RepAnalysis) -> tuple[str, float, list[str]]:
    lockout_quality = _metric(analysis, "avg_lockout_quality")
    wrist_travel = _metric(analysis, "avg_wrist_travel")
    wrist_asymmetry = _metric(analysis, "avg_wrist_height_asymmetry")

    if wrist_asymmetry is not None and wrist_asymmetry >= 0.12:
        return (
            "asymmetric_press",
            _confidence(0.76, analysis, wrist_asymmetry),
            ["intentional_asymmetry_check"],
        )
    if (lockout_quality is not None and lockout_quality <= 0.65) or (
        wrist_travel is not None and wrist_travel < 0.24
    ):
        support = 1.0 - lockout_quality if lockout_quality is not None else wrist_travel
        return "partial_press", _confidence(0.72, analysis, support), ["partial_range_of_motion"]

    not_issues = ["lockout_unverified"] if lockout_quality is None else []
    return "standing_shoulder_press", _confidence(0.62, analysis, lockout_quality), not_issues


def _detected_variation(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
) -> tuple[str, float, list[str]]:
    if classification.exercise == "push_up":
        return _push_up_variation(analysis)
    if classification.exercise == "squat":
        return _squat_variation(analysis)
    if classification.exercise == "shoulder_press":
        return _shoulder_press_variation(analysis)

    exercise_spec = get_exercise_spec(classification.exercise)
    return (
        exercise_spec.default_variation,
        exercise_spec.default_variation_confidence,
        list(exercise_spec.default_variation_not_issues),
    )


def _profile_not_issues(variation: str) -> list[str]:
    mapping: dict[str, list[str]] = {
        "wide_grip_push_up": ["wide_hand_placement"],
        "close_grip_push_up": ["close_hand_placement"],
        "knee_push_up": ["knee_contact"],
        "wide_squat_stance": ["wide_stance"],
        "narrow_squat_stance": ["narrow_stance"],
        "pause_squat": ["bottom_pause"],
        "partial_press": ["partial_range_of_motion"],
        "asymmetric_press": ["intentional_asymmetry_check"],
    }
    return list(mapping.get(variation, []))


def run(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    profile: UserProfile,
) -> Variation:
    if profile.intended_variation:
        variation = profile.intended_variation
        confidence = 0.95
        not_issues = _profile_not_issues(variation)
    else:
        variation, confidence, not_issues = _detected_variation(classification, analysis)

    if analysis.aggregate_metrics.get("avg_rom_score", 1.0) < 0.7:
        not_issues.append("low_rom_requires_user_intent_check")
    if analysis.aggregate_metrics.get("pose_valid_ratio", 1.0) < 0.8:
        not_issues.append("low_pose_confidence_limits_variation_call")

    return Variation(
        exercise=classification.exercise,
        detected_variation=variation,
        variation_confidence=confidence,
        not_issues=sorted(set(not_issues)),
    )
