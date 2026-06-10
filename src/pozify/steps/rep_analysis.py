from __future__ import annotations

from statistics import mean

from pozify.contracts import (
    ExerciseClassification,
    PoseSequence,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)


def _metrics_for_exercise(exercise: str, rep_id: int) -> dict[str, float | str]:
    fatigue_penalty = max(0, rep_id - 3) * 0.08
    if exercise == "squat":
        return {
            "min_knee_angle_deg": 92 - rep_id,
            "hip_depth_relative_to_knee": "slightly_above_parallel" if rep_id >= 4 else "parallel",
            "max_torso_lean_deg": 28 + rep_id,
            "knee_tracking_score": round(0.84 - fatigue_penalty, 2),
        }
    if exercise == "shoulder_press":
        return {
            "min_elbow_angle_deg": 74 + rep_id,
            "lockout_quality": round(0.9 - fatigue_penalty, 2),
            "wrist_path_verticality": round(0.86 - fatigue_penalty, 2),
            "left_right_wrist_delta": round(0.02 + fatigue_penalty, 2),
        }
    return {
        "min_elbow_angle_deg": 88 + rep_id,
        "body_line_score": round(0.9 - fatigue_penalty, 2),
        "hip_sag_score": round(0.18 + fatigue_penalty, 2),
        "hand_width_ratio": 1.42,
    }


def run(
    classification: ExerciseClassification,
    reps: Reps,
    sequence: PoseSequence,
) -> RepAnalysis:
    items: list[RepAnalysisItem] = []
    for rep in reps.reps:
        fatigue_penalty = max(0, rep.rep_id - 3) * 0.08
        items.append(
            RepAnalysisItem(
                rep_id=rep.rep_id,
                duration_sec=round(rep.end_sec - rep.start_sec, 2),
                range_of_motion_score=round(0.88 - fatigue_penalty, 2),
                stability_score=round(0.86 - fatigue_penalty, 2),
                symmetry_score=round(0.9 - fatigue_penalty / 2, 2),
                metrics=_metrics_for_exercise(classification.exercise, rep.rep_id),
                variation_hints=["wide_grip_push_up"]
                if classification.exercise == "push_up"
                else [],
            )
        )

    return RepAnalysis(
        exercise=classification.exercise,
        items=items,
        aggregate_metrics={
            "avg_rom_score": round(mean(item.range_of_motion_score for item in items), 2),
            "avg_stability_score": round(mean(item.stability_score for item in items), 2),
            "avg_symmetry_score": round(mean(item.symmetry_score for item in items), 2),
            "pose_valid_ratio": sequence.pose_valid_ratio,
            "mock": True,
        },
    )

