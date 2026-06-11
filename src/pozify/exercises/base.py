from __future__ import annotations

from pozify.contracts import PoseFrame, RepAnalysis
from pozify.exercises.shared.analyzer import ExerciseMetricResult
from pozify.exercises.shared.issue_marker import IssueRule
from pozify.steps.rep_counters.base import ExerciseRepCounter


class ExerciseStrategy(ExerciseRepCounter):
    def metrics(self, frames: list[PoseFrame]) -> ExerciseMetricResult:
        raise NotImplementedError

    def detect_variation(self, analysis: RepAnalysis) -> tuple[str, float, list[str]]:
        raise NotImplementedError

    def profile_not_issues(self, variation: str) -> list[str]:
        return []

    def issue_rules(self) -> tuple[IssueRule, ...]:
        return ()

    def metric(self, analysis: RepAnalysis, name: str) -> float | None:
        value = analysis.aggregate_metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def confidence(
        self,
        base: float,
        analysis: RepAnalysis,
        supporting_metric: float | None,
    ) -> float:
        rep_bonus = min(0.12, len(analysis.items) * 0.03)
        metric_bonus = 0.0 if supporting_metric is None else min(0.1, abs(supporting_metric) * 0.03)
        pose_bonus = min(0.08, float(analysis.aggregate_metrics.get("pose_valid_ratio", 0.0)) * 0.08)
        return round(min(0.95, base + rep_bonus + metric_bonus + pose_bonus), 2)
