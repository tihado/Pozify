from __future__ import annotations

from pozify.contracts import ExerciseClassification, PoseSequence, Rep, Reps


def run(classification: ExerciseClassification, sequence: PoseSequence) -> Reps:
    fps = 30.0
    reps: list[Rep] = []
    for idx in range(5):
        start_frame = 30 + idx * 60
        mid_frame = start_frame + 30
        end_frame = start_frame + 60
        reps.append(
            Rep(
                rep_id=idx + 1,
                start_frame=start_frame,
                mid_frame=mid_frame,
                end_frame=end_frame,
                start_sec=round(start_frame / fps, 2),
                mid_sec=round(mid_frame / fps, 2),
                end_sec=round(end_frame / fps, 2),
            )
        )

    return Reps(
        exercise=classification.exercise,
        reps=reps,
        partial_reps=[] if sequence.pose_valid_ratio > 0.8 else [{"reason": "low_pose_valid_ratio"}],
    )

