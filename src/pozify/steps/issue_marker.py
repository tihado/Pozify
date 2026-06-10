from __future__ import annotations

from pozify.contracts import (
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    RepAnalysis,
    Reps,
    Variation,
)
from pozify.exercise_catalog import get_exercise_spec


def run(
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
) -> IssueMarkers:
    issues: list[IssueMarker] = []
    exercise_spec = get_exercise_spec(classification.exercise)
    for item in analysis.items:
        if item.stability_score >= 0.78 or exercise_spec.mock_issue is None:
            continue

        rep = next(rep for rep in reps.reps if rep.rep_id == item.rep_id)
        issue_spec = exercise_spec.mock_issue
        metric_value = (
            item.range_of_motion_score
            if issue_spec.evidence_metric == "range_of_motion_score"
            else item.metrics.get(issue_spec.evidence_metric)
        )
        evidence = {
            issue_spec.evidence_metric: metric_value,
            "threshold": issue_spec.threshold,
            "variation": variation.detected_variation,
        }

        issues.append(
            IssueMarker(
                rep_id=item.rep_id,
                issue=issue_spec.issue,
                severity=round(1.0 - item.stability_score, 2),
                start_frame=rep.mid_frame,
                end_frame=rep.end_frame,
                start_sec=rep.mid_sec,
                end_sec=rep.end_sec,
                affected_joints=list(issue_spec.affected_joints),
                evidence=evidence,
            )
        )

    return IssueMarkers(issues=issues)
