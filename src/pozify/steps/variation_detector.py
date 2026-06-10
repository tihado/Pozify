from __future__ import annotations

from pozify.contracts import ExerciseClassification, RepAnalysis, UserProfile, Variation


def run(
    classification: ExerciseClassification,
    analysis: RepAnalysis,
    profile: UserProfile,
) -> Variation:
    if profile.intended_variation:
        variation = profile.intended_variation
        confidence = 0.95
    elif classification.exercise == "push_up":
        variation = "wide_grip_push_up"
        confidence = 0.84
    elif classification.exercise == "squat":
        variation = "bodyweight_squat"
        confidence = 0.82
    elif classification.exercise == "shoulder_press":
        variation = "standing_shoulder_press"
        confidence = 0.8
    else:
        variation = "unknown"
        confidence = 0.3

    not_issues = ["wide_hand_placement"] if variation == "wide_grip_push_up" else []
    if analysis.aggregate_metrics.get("avg_rom_score", 1.0) < 0.7:
        not_issues.append("mock_low_rom_requires_user_intent_check")

    return Variation(
        exercise=classification.exercise,
        detected_variation=variation,
        variation_confidence=confidence,
        not_issues=not_issues,
    )

