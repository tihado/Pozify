from __future__ import annotations

from pozify.contracts import (
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    RepAnalysis,
    Reps,
    Variation,
)


def run(
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
) -> IssueMarkers:
    issues: list[IssueMarker] = []
    for item in analysis.items:
        if item.stability_score >= 0.78:
            continue

        rep = next(rep for rep in reps.reps if rep.rep_id == item.rep_id)
        if classification.exercise == "squat":
            issue = "shallow_depth"
            affected_joints = ["left_hip", "right_hip", "left_knee", "right_knee"]
            evidence = {
                "range_of_motion_score": item.range_of_motion_score,
                "threshold": 0.8,
            }
        elif classification.exercise == "shoulder_press":
            issue = "incomplete_lockout"
            affected_joints = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
            evidence = {
                "lockout_quality": item.metrics.get("lockout_quality"),
                "threshold": 0.8,
            }
        else:
            issue = "hip_sag"
            affected_joints = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
            evidence = {
                "body_line_score": item.metrics.get("body_line_score"),
                "threshold": 0.8,
                "variation": variation.detected_variation,
            }

        issues.append(
            IssueMarker(
                rep_id=item.rep_id,
                issue=issue,
                severity=round(1.0 - item.stability_score, 2),
                start_frame=rep.mid_frame,
                end_frame=rep.end_frame,
                start_sec=rep.mid_sec,
                end_sec=rep.end_sec,
                affected_joints=affected_joints,
                evidence=evidence,
            )
        )

    return IssueMarkers(issues=issues)

