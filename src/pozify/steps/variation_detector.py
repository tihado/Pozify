from __future__ import annotations

from pozify.contracts import ExerciseClassification, RepAnalysis, UserProfile, Variation
from pozify.exercise_catalog import get_exercise_spec


def run(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    profile: UserProfile,
) -> Variation:
    exercise_spec = get_exercise_spec(classification.exercise)
    if profile.intended_variation:
        variation = profile.intended_variation
        confidence = 0.95
    else:
        variation = exercise_spec.default_variation
        confidence = exercise_spec.default_variation_confidence

    not_issues = (
        list(exercise_spec.default_variation_not_issues)
        if variation == exercise_spec.default_variation
        else []
    )
    if analysis.aggregate_metrics.get("avg_rom_score", 1.0) < 0.7:
        not_issues.append("mock_low_rom_requires_user_intent_check")

    return Variation(
        exercise=classification.exercise,
        detected_variation=variation,
        variation_confidence=confidence,
        not_issues=not_issues,
    )
