from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import CoachSummary, ExerciseClassification, IssueMarker, IssueMarkers, Variation
from pozify.steps import verifier


class VerifierTests(unittest.TestCase):
    def test_rejects_issue_not_present_in_json(self) -> None:
        summary = CoachSummary(
            summary="This report shows knee_valgus and should prevent injury.",
            what_went_well=["Tempo looked steady."],
            main_findings=["knee_valgus appeared in several reps."],
            variation_explanation="`wide_grip_push_up` is an error.",
            top_fixes=["Fix the injury risk now."],
            next_session_plan=["Change everything next session."],
            confidence_notes=[],
        )
        verification = verifier.run(
            summary,
            IssueMarkers([IssueMarker(1, "hip_sag", 0.3, 0, 10, 0.0, 0.33, [], {})]),
            Variation("push_up", "wide_grip_push_up", 0.84, ["wide_hand_placement"]),
            ExerciseClassification("push_up", 0.92, [], False),
            mock_steps=["exercise_classifier"],
        )
        self.assertFalse(verification.passed)
        self.assertIn("absent from issue_markers.json", " ".join(verification.notes))


if __name__ == "__main__":
    unittest.main()
