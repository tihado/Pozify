from __future__ import annotations

from statistics import mean
from typing import Any

from pozify.contracts import (
    ExerciseClassification,
    PoseSequence,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
)
from pozify.exercise_catalog import get_exercise_spec


def _metrics_for_exercise(exercise: str, rep_id: int) -> dict[str, Any]:
    fatigue_penalty = max(0, rep_id - 3) * 0.08
    return get_exercise_spec(exercise).metric_factory(rep_id, fatigue_penalty)


def run(
    classification: ExerciseClassification,
    reps: Reps,
    sequence: PoseSequence,
) -> RepAnalysis:
    items: list[RepAnalysisItem] = []
    exercise_spec = get_exercise_spec(classification.exercise)
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
                variation_hints=list(exercise_spec.variation_hints),
            )
        )

    aggregate_metrics = {
        "avg_rom_score": round(mean(item.range_of_motion_score for item in items), 2) if items else 0.0,
        "avg_stability_score": round(mean(item.stability_score for item in items), 2) if items else 0.0,
        "avg_symmetry_score": round(mean(item.symmetry_score for item in items), 2) if items else 0.0,
        "pose_valid_ratio": sequence.pose_valid_ratio,
        "mock": True,
    }

    return RepAnalysis(
        exercise=classification.exercise,
        items=items,
        aggregate_metrics=aggregate_metrics,
    )
