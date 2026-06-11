from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
from pozify.steps import coach_summary, verifier


class CoachSummaryTests(unittest.TestCase):
    def _inputs(self) -> tuple[UserProfile, ExerciseClassification, Reps, RepAnalysis, Variation, IssueMarkers]:
        return (
            UserProfile("beginner_practice", "beginner"),
            ExerciseClassification("push_up", 0.92, [], False),
            Reps("push_up", [Rep(1, 0, 10, 20, 0.0, 0.33, 0.67)], []),
            RepAnalysis(
                "push_up",
                [RepAnalysisItem(1, 0.67, 0.82, 0.8, 0.84, {"body_line_score": 0.78}, ["wide_grip_push_up"])],
                {"avg_rom_score": 0.82, "avg_stability_score": 0.8, "avg_symmetry_score": 0.84, "pose_valid_ratio": 1.0},
            ),
            Variation("push_up", "wide_grip_push_up", 0.84, ["wide_hand_placement"]),
            IssueMarkers([IssueMarker(1, "hip_sag", 0.3, 10, 20, 0.33, 0.67, ["left_hip"], {"body_line_score": 0.78})]),
        )

    def test_template_summary_is_grounded(self) -> None:
        profile, classification, reps, analysis, variation, issues = self._inputs()
        summary = coach_summary.run(
            profile, classification, reps, analysis, variation, issues, mock_steps=[]
        )
        self.assertIn("wide_grip_push_up", summary.variation_explanation)
        self.assertTrue(summary.confidence_notes)
        verification = verifier.run(summary, issues, variation, classification, mock_steps=[])
        self.assertTrue(verification.passed)

    def test_pipeline_can_fallback_from_unsafe_provider(self) -> None:
        profile, classification, reps, analysis, variation, issues = self._inputs()
        with patch.dict("os.environ", {"POZIFY_SUMMARY_PROVIDER": "unsafe_mock"}):
            draft = coach_summary.run(
                profile, classification, reps, analysis, variation, issues, mock_steps=[]
            )
        verification = verifier.run(draft, issues, variation, classification, mock_steps=[])
        self.assertFalse(verification.passed)
        fallback = coach_summary.build_fallback(
            profile,
            classification,
            reps,
            analysis,
            variation,
            issues,
            verification_notes=verification.notes,
            mock_steps=[],
        )
        self.assertIn("fallback summary", " ".join(fallback.main_findings).lower())

    def test_generate_returns_parse_failure_metadata_for_invalid_slm_output(self) -> None:
        profile, classification, reps, analysis, variation, issues = self._inputs()
        with patch.dict("os.environ", {"POZIFY_SUMMARY_PROVIDER": "slm_local"}):
            with patch(
                "pozify.steps.summary_provider.create_summary_slm_backend"
            ) as create_backend:
                create_backend.return_value = type(
                    "FakeBackend",
                    (),
                    {
                        "backend_name": "transformers",
                        "model_name": "Qwen/Fake",
                        "generate_text": lambda self, prompt: type(
                            "BackendResult",
                            (),
                            {
                                "text": "not valid json",
                                "backend": "transformers",
                                "model": "Qwen/Fake",
                            },
                        )(),
                    },
                )()
                draft = coach_summary.generate(
                    profile, classification, reps, analysis, variation, issues, mock_steps=[]
                )

        self.assertIsNone(draft.summary)
        self.assertFalse(draft.generation.parse_ok)
        self.assertEqual(draft.generation.provider, "slm_local")


if __name__ == "__main__":
    unittest.main()
