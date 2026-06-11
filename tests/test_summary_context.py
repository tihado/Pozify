from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import (
    ExerciseClassification,
    IssueMarker,
    IssueMarkers,
    Rep,
    RepAnalysis,
    RepAnalysisItem,
    Reps,
    UserProfile,
    Variation,
)
from pozify.steps.summary_context import build_summary_context


class SummaryContextTests(unittest.TestCase):
    def test_builds_grounded_context_with_cards(self) -> None:
        context = build_summary_context(
            UserProfile("strength", "beginner"),
            ExerciseClassification("squat", 0.92, [], False),
            Reps("squat", [Rep(1, 0, 10, 20, 0.0, 0.33, 0.67)], []),
            RepAnalysis(
                "squat",
                [RepAnalysisItem(1, 0.67, 0.8, 0.82, 0.84, {"tempo_consistency_score": 0.9}, [])],
                {"avg_rom_score": 0.8, "avg_stability_score": 0.82, "avg_symmetry_score": 0.84},
            ),
            Variation("squat", "bodyweight_squat", 0.82, []),
            IssueMarkers(
                [IssueMarker(1, "shallow_depth", 0.4, 10, 20, 0.33, 0.67, ["left_knee"], {"threshold": 0.8})]
            ),
            mock_steps=["exercise_classifier"],
        )
        self.assertEqual(context["exercise"]["label"], "squat")
        self.assertTrue(context["constraints"]["must_not_invent_issues"])
        self.assertGreaterEqual(len(context["knowledge_cards"]), 4)


if __name__ == "__main__":
    unittest.main()
