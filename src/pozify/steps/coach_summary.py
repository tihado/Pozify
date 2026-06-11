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
        f"The strongest frame-level marker is `{main_issue}`, with timestamps and metric evidence in the issue timeline."
        if issues.issues
        else "The frame-level issue rules did not find a sustained threshold violation."
    )

    return CoachSummary(
        summary=(
            f"The pipeline segmented {len(reps.reps)} {classification.exercise} reps from the current pose stream. "
            f"Exercise routing, variation labeling, and coaching language may still use lightweight rules or placeholders. "
            f"The detected variation is `{variation.detected_variation}`. {issue_text}"
        ),
        what_went_well=[
            f"Current placeholder ROM score is {analysis.aggregate_metrics['avg_rom_score']}.",
            f"Current placeholder symmetry score is {analysis.aggregate_metrics['avg_symmetry_score']}.",
        ],
        main_findings=[
            f"Frame-level marker `{main_issue}` appears in {len(issues.issues)} interval(s)."
            if issues.issues
            else "No sustained issue interval detected."
        ],
        variation_explanation=(
            f"`{variation.detected_variation}` was included as context for interpreting issue markers. "
            f"It should be reviewed alongside camera angle and pose confidence."
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
            f"Current mock classifier confidence placeholder: {classification.confidence:.0%}.",
            "Rep segmentation can run on real pose, but downstream interpretation is still partially mocked.",
        ],
    )
