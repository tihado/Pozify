from __future__ import annotations

from pozify.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
)
from pozify.steps.summary_context import build_summary_context
from pozify.steps.summary_provider import create_summary_provider


def run(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    mock_steps: list[str] | None = None,
) -> CoachSummary:
    context = build_summary_context(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        mock_steps=mock_steps,
    )
    payload = create_summary_provider().generate(context)
    return CoachSummary(
        summary=payload["summary"],
        what_went_well=list(payload["what_went_well"]),
        main_findings=list(payload["main_findings"]),
        variation_explanation=payload["variation_explanation"],
        top_fixes=list(payload["top_fixes"]),
        next_session_plan=list(payload["next_session_plan"]),
        confidence_notes=list(payload["confidence_notes"]),
    )


def build_fallback(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    verification_notes: list[str],
    mock_steps: list[str] | None = None,
) -> CoachSummary:
    issue_labels = sorted({issue.issue for issue in issues.issues})
    issue_text = (
        "Observed issue labels: " + ", ".join(f"`{label}`" for label in issue_labels) + "."
        if issue_labels
        else "No issue labels were emitted from the current issue marker step."
    )
    return CoachSummary(
        summary=(
            f"The pipeline observed {len(reps.reps)} {classification.exercise} rep(s) and the variation label "
            f"`{variation.detected_variation}`. {issue_text}"
        ),
        what_went_well=[
            f"Average ROM score from the structured artifacts is {analysis.aggregate_metrics['avg_rom_score']}.",
            f"Average symmetry score from the structured artifacts is {analysis.aggregate_metrics['avg_symmetry_score']}.",
        ],
        main_findings=[
            issue_text,
            "A conservative fallback summary was used because the generated draft failed verification.",
        ],
        variation_explanation=(
            f"`{variation.detected_variation}` is treated as context and not automatically as an error."
        ),
        top_fixes=[
            "Keep the camera angle and setup consistent for the next review.",
            "Focus on one visible adjustment at a time.",
            "Use the issue timeline and aggregate metrics as the main comparison points.",
        ],
        next_session_plan=[
            "Repeat the same recording setup on the next set.",
            "Compare whether the same issue labels appear in the same part of the set.",
            "Use a slower tempo if you want cleaner evidence for the next comparison.",
        ],
        confidence_notes=[
            *verification_notes,
            f"Classifier confidence for this run is {classification.confidence:.0%}.",
            (
                "Some steps still use placeholders: " + ", ".join(mock_steps)
                if mock_steps
                else "Fallback summary stayed within structured evidence only."
            ),
        ],
    )
