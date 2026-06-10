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


def run(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
) -> CoachSummary:
    issue_labels = [issue.issue for issue in issues.issues]
    main_issue = issue_labels[0] if issue_labels else "no_major_issue"
    issue_text = (
        f"The main mocked finding is `{main_issue}`, mostly appearing in later reps."
        if issues.issues
        else "The mocked analysis did not detect a major issue."
    )

    return CoachSummary(
        summary=(
            f"You completed {len(reps.reps)} mocked {classification.exercise} reps. "
            f"The detected variation is `{variation.detected_variation}`. {issue_text} "
            f"This feedback is practice-oriented and based on structured mock metrics."
        ),
        what_went_well=[
            f"Average range of motion score is {analysis.aggregate_metrics['avg_rom_score']}.",
            f"Average symmetry score is {analysis.aggregate_metrics['avg_symmetry_score']}.",
        ],
        main_findings=[
            f"{main_issue} detected in {len(issues.issues)} rep(s)."
            if issues.issues
            else "No major mocked issue detected."
        ],
        variation_explanation=(
            f"`{variation.detected_variation}` is treated as the selected/detected variation. "
            f"These labels are kept separate from true issue labels."
        ),
        top_fixes=[
            "Keep the same setup and record from a stable camera angle.",
            "Stop the set when the first repeated issue marker appears.",
            "Use slower reps until the highlighted metric improves.",
        ],
        next_session_plan=[
            "Warm up with 1 easy set of 5 controlled reps.",
            "Perform 2 working sets and keep the camera angle consistent.",
            "Compare the next run against this report's issue timeline.",
        ],
        confidence_notes=[
            f"Exercise classifier confidence: {classification.confidence:.0%}.",
            "This is mock output; replace step implementations before using as real coaching.",
        ],
    )

