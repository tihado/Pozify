from __future__ import annotations

from pozify.contracts import RepAnalysis, UserProfile, Variation
from pozify.exercises import ExerciseStrategy


def run(
    exercise: ExerciseStrategy,
    analysis: RepAnalysis,
    profile: UserProfile,
) -> Variation:
    if profile.intended_variation:
        variation = profile.intended_variation
        confidence = 0.95
        not_issues = exercise.profile_not_issues(variation)
    else:
        variation, confidence, not_issues = exercise.detect_variation(analysis)

    if analysis.aggregate_metrics.get("avg_rom_score", 1.0) < 0.7:
        not_issues.append("low_rom_requires_user_intent_check")
    if analysis.aggregate_metrics.get("pose_valid_ratio", 1.0) < 0.8:
        not_issues.append("low_pose_confidence_limits_variation_call")

    return Variation(
        exercise=exercise.exercise,
        detected_variation=variation,
        variation_confidence=confidence,
        not_issues=sorted(set(not_issues)),
    )
