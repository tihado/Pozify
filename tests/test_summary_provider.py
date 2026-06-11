from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.steps.summary_provider import OpenSourceSlmProvider, build_prompt_contract
from pozify.steps.summary_slm_backend import (
    LlamaCppGgufSummaryBackend,
    create_summary_slm_backend,
)


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

    def test_backend_factory_supports_gguf(self) -> None:
        with patch.dict("os.environ", {"POZIFY_SUMMARY_BACKEND": "gguf"}, clear=False):
            backend = create_summary_slm_backend()
        self.assertIsInstance(backend, LlamaCppGgufSummaryBackend)

    def test_gguf_backend_generates_text_via_llama_cpp(self) -> None:
        class FakeLlama:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def create_chat_completion(self, messages, max_tokens, temperature):
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert max_tokens == 128
                assert temperature == 0.0
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"OK","what_went_well":["A"],"main_findings":["B"],'
                                    '"variation_explanation":"C","top_fixes":["D"],'
                                    '"next_session_plan":["E"],"confidence_notes":["F"]}'
                                )
                            }
                        }
                    ]
                }

        fake_llama_cpp = types.SimpleNamespace(Llama=FakeLlama)

        with patch.dict(
            "sys.modules",
            {"llama_cpp": fake_llama_cpp},
        ):
            with patch.object(
                LlamaCppGgufSummaryBackend,
                "_resolve_model_path",
                return_value="/tmp/qwen.gguf",
            ):
                with patch.dict(
                    "os.environ",
                    {
                        "POZIFY_SUMMARY_BACKEND": "gguf",
                        "POZIFY_SUMMARY_MAX_TOKENS": "128",
                        "POZIFY_SUMMARY_TEMPERATURE": "0",
                    },
                    clear=False,
                ):
                    result = OpenSourceSlmProvider().generate(
                        {
                            "user_profile": {},
                            "exercise": {},
                            "rep_summary": {},
                            "variation": {},
                            "issues": [],
                            "knowledge_cards": [],
                            "retrieval_trace": {
                                "missing_labels": [],
                                "matched_card_ids": [],
                                "requested_labels": [],
                            },
                            "constraints": {},
                            "mock_steps": [],
                        }
                    )

        self.assertTrue(result.parse_ok)
        self.assertEqual(result.backend, "gguf")
        self.assertIn(".gguf", result.model or "")


if __name__ == "__main__":
    unittest.main()
