from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.steps.summary_provider import OpenSourceSlmProvider, build_prompt_contract


class SummaryProviderTests(unittest.TestCase):
    def test_prompt_contract_requests_json_only(self) -> None:
        prompt = build_prompt_contract(
            {
                "user_profile": {},
                "exercise": {},
                "rep_summary": {},
                "variation": {},
                "issues": [],
                "knowledge_cards": [],
                "retrieval_trace": {"missing_labels": [], "matched_card_ids": [], "requested_labels": []},
                "constraints": {},
                "mock_steps": [],
            }
        )
        self.assertIn("Return valid JSON only", prompt)

    def test_slm_provider_parses_valid_json_from_backend(self) -> None:
        class FakeBackend:
            backend_name = "transformers"
            model_name = "Qwen/Fake"

            def generate_text(self, prompt: str) -> object:
                del prompt
                return type(
                    "BackendResult",
                    (),
                    {
                        "text": (
                            '{"summary":"OK","what_went_well":["A"],"main_findings":["B"],'
                            '"variation_explanation":"C","top_fixes":["D"],'
                            '"next_session_plan":["E"],"confidence_notes":["F"]}'
                        ),
                        "backend": "transformers",
                        "model": "Qwen/Fake",
                    },
                )()

        with patch("pozify.steps.summary_provider.create_summary_slm_backend", return_value=FakeBackend()):
            result = OpenSourceSlmProvider().generate(
                {
                    "user_profile": {},
                    "exercise": {},
                    "rep_summary": {},
                    "variation": {},
                    "issues": [],
                    "knowledge_cards": [],
                    "retrieval_trace": {"missing_labels": [], "matched_card_ids": [], "requested_labels": []},
                    "constraints": {},
                    "mock_steps": [],
                }
            )

        self.assertTrue(result.parse_ok)
        self.assertIsNotNone(result.payload)
        assert result.payload is not None
        self.assertEqual(result.payload["summary"], "OK")
        self.assertEqual(result.provider, "slm_local")

    def test_slm_provider_reports_parse_failure(self) -> None:
        class FakeBackend:
            backend_name = "transformers"
            model_name = "Qwen/Fake"

            def generate_text(self, prompt: str) -> object:
                del prompt
                return type(
                    "BackendResult",
                    (),
                    {
                        "text": "not json at all",
                        "backend": "transformers",
                        "model": "Qwen/Fake",
                    },
                )()

        with patch("pozify.steps.summary_provider.create_summary_slm_backend", return_value=FakeBackend()):
            result = OpenSourceSlmProvider().generate(
                {
                    "user_profile": {},
                    "exercise": {},
                    "rep_summary": {},
                    "variation": {},
                    "issues": [],
                    "knowledge_cards": [],
                    "retrieval_trace": {"missing_labels": [], "matched_card_ids": [], "requested_labels": []},
                    "constraints": {},
                    "mock_steps": [],
                }
            )

        self.assertFalse(result.parse_ok)
        self.assertIsNone(result.payload)
        self.assertIn("JSON object", result.parse_error or "")


if __name__ == "__main__":
    unittest.main()
