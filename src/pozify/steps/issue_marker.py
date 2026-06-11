from __future__ import annotations

from pozify.contracts import (
    IssueMarker,
    IssueMarkers,
    PoseSequence,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
    Variation,
)
from pozify.exercise_catalog import get_exercise_spec
from pozify.exercises import ExerciseStrategy
from pozify.exercises.shared.issue_marker import (
    frame_scores_for_rule,
    frames_for_rep,
    group_violations,
    marker_from_group,
    minimum_run_length,
)


def _fallback_rep_marker(
    reps: Reps,
    item: RepAnalysisItem,
    variation: Variation,
) -> IssueMarker | None:
    exercise_spec = get_exercise_spec(reps.exercise)
    if item.stability_score >= 0.78 or exercise_spec.mock_issue is None:
        return None

    rep = next((rep for rep in reps.reps if rep.rep_id == item.rep_id), None)
    if rep is None:
        return None

    issue_spec = exercise_spec.mock_issue
    metric_value = (
        item.range_of_motion_score
        if issue_spec.evidence_metric == "range_of_motion_score"
        else item.metrics.get(issue_spec.evidence_metric)
    )
    return IssueMarker(
        rep_id=item.rep_id,
        issue=issue_spec.issue,
        severity=round(1.0 - item.stability_score, 2),
        start_frame=rep.mid_frame,
        end_frame=rep.end_frame,
        start_sec=rep.mid_sec,
        end_sec=rep.end_sec,
        affected_joints=list(issue_spec.affected_joints),
        evidence={
            issue_spec.evidence_metric: metric_value,
            "threshold": issue_spec.threshold,
            "confidence": round(max(0.0, min(1.0, 1.0 - item.stability_score)), 2),
            "variation_context": {
                "detected_variation": variation.detected_variation,
                "variation_confidence": variation.variation_confidence,
                "not_issues": list(variation.not_issues),
            },
            "fallback": "rep_level_metrics",
        },
    )


def run(
    exercise: ExerciseStrategy,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    sequence: PoseSequence | None = None,
) -> IssueMarkers:
    rep_by_id = {rep.rep_id: rep for rep in reps.reps}
    rules = exercise.issue_rules()
    issues: list[IssueMarker] = []

    for item in analysis.items:
        rep = rep_by_id.get(item.rep_id)
        if rep is None:
            continue

        frames = frames_for_rep(sequence, rep)
        if not frames:
            fallback = _fallback_rep_marker(reps, item, variation)
            if fallback is not None:
                issues.append(fallback)
            continue

        min_run_length = minimum_run_length(frames)
        for rule in rules:
            if set(rule.suppress_when_not_issue) & set(variation.not_issues):
                continue

            scores = frame_scores_for_rule(frames, exercise.exercise, rule)
            for group in group_violations(scores, min_run_length):
                issues.append(marker_from_group(rule, group, item, variation))

    return IssueMarkers(
        issues=sorted(
            issues,
            key=lambda issue: (issue.start_frame, issue.end_frame, issue.rep_id, issue.issue),
        )
    )
