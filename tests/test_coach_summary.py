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
from pozify.steps.summary_provider import SummaryProviderResult


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
        with patch.dict("os.environ", {"POZIFY_SUMMARY_PROVIDER": "slm_cloud"}):
            with patch(
                "pozify.steps.summary_provider.HuggingFaceCloudSummaryProvider.generate"
            ) as generate:
                generate.return_value = SummaryProviderResult(
                    payload=None,
                    provider="slm_cloud",
                    backend="huggingface",
                    model="Qwen/Fake",
                    prompt_contract_version="v1",
                    parse_ok=False,
                    parse_error="not valid json",
                )
                draft = coach_summary.generate(
                    profile, classification, reps, analysis, variation, issues, mock_steps=[]
                )

        self.assertIsNone(draft.summary)
        self.assertFalse(draft.generation.parse_ok)
        self.assertEqual(draft.generation.provider, "slm_cloud")

    def test_fallback_hides_internal_provider_error_from_confidence_notes(self) -> None:
        profile, classification, reps, analysis, variation, issues = self._inputs()
        fallback = coach_summary.build_fallback(
            profile,
            classification,
            reps,
            analysis,
            variation,
            issues,
            verification_notes=[
                "Summary provider failed before verification.",
                "The Hugging Face cloud summary provider requires HF_TOKEN or POZIFY_SUMMARY_API_KEY.",
                "Conservative fallback summary returned.",
            ],
            mock_steps=[],
        )

        joined = " ".join(fallback.confidence_notes).lower()
        self.assertNotIn("hf_token", joined)
        self.assertNotIn("provider failed before verification", joined)
        self.assertIn("fallback summary", joined)

    def test_fallback_hides_mps_out_of_memory_details_from_confidence_notes(self) -> None:
        profile, classification, reps, analysis, variation, issues = self._inputs()
        fallback = coach_summary.build_fallback(
            profile,
            classification,
            reps,
            analysis,
            variation,
            issues,
            verification_notes=[
                "Summary provider failed before verification.",
                "MPS backend out of memory (MPS allocated: 6.70 GiB, other allocations: 5.70 GiB, max allowed: 9.07 GiB). Tried to allocate 26.79 MiB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).",
                "Conservative fallback summary returned.",
            ],
            mock_steps=[],
        )

        joined = " ".join(fallback.confidence_notes).lower()
        self.assertNotIn("out of memory", joined)
        self.assertNotIn("mps backend", joined)
        self.assertNotIn("tried to allocate", joined)
        self.assertIn("fallback summary", joined)


if __name__ == "__main__":
    unittest.main()
