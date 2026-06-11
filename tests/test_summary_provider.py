from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.steps.summary_provider import (
    CoachSummaryPayloadModel,
    HuggingFaceCloudSummaryProvider,
    _structured_response_format,
    build_prompt_contract,
)
from pozify.steps import summary_provider


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

    def test_prepare_context_for_cloud_truncates_large_context(self) -> None:
        context = {
            "user_profile": {},
            "exercise": {"label": "squat", "confidence": 0.99},
            "rep_summary": {"rep_count": 15, "partial_reps": [], "aggregate_metrics": {}, "trends": {}},
            "variation": {"label": "wide_squat_stance", "confidence": 0.95, "not_issues": ["wide_stance"]},
            "issues": [
                {
                    "issue": "shallow_depth",
                    "rep_id": index,
                    "severity": 0.9 - index * 0.01,
                    "start_sec": 1.0,
                    "end_sec": 1.3,
                    "affected_joints": ["left_hip", "right_hip", "left_knee", "right_knee"],
                    "evidence": {
                        "confidence": 0.7,
                        "threshold": -0.03,
                        "mean_metric_value": -0.05,
                        "supporting_frames": 12,
                        "large_blob": "x" * 400,
                    },
                }
                for index in range(20)
            ],
            "knowledge_cards": [
                {
                    "id": f"card-{index}",
                    "type": "issue",
                    "label": f"issue_{index}",
                    "summary": "summary " + ("y" * 300),
                    "good_signals": ["good"] * 4,
                    "common_misreads": ["misread"] * 4,
                    "coaching_cues": ["cue"] * 4,
                    "safety_notes": ["safe"] * 4,
                    "contraindicated_claims": ["claim"] * 4,
                }
                for index in range(12)
            ],
            "retrieval_trace": {
                "requested_labels": ["squat", "wide_squat_stance"],
                "matched_card_ids": [f"card-{index}" for index in range(12)],
                "missing_labels": [],
            },
            "constraints": {},
            "mock_steps": [],
        }

        prepared, trimmed = summary_provider._prepare_context_for_backend(
            context,
            backend_name="huggingface",
            context_window=2048,
        )

        self.assertTrue(trimmed)
        self.assertLessEqual(len(prepared["issues"]), 12)
        self.assertLessEqual(len(prepared["knowledge_cards"]), 8)
        self.assertIn("issue_overview", prepared)

    def test_build_prompt_contract_compact_is_shorter(self) -> None:
        context = {
            "user_profile": {"goal": "beginner_practice"},
            "exercise": {"label": "squat"},
            "rep_summary": {"rep_count": 10},
            "variation": {"label": "wide_squat_stance"},
            "issues": [{"issue": "shallow_depth", "rep_id": 1}],
            "knowledge_cards": [{"id": "1", "type": "issue", "label": "shallow_depth"}],
            "retrieval_trace": {"missing_labels": [], "matched_card_ids": [], "requested_labels": []},
            "constraints": {},
            "mock_steps": [],
        }
        standard = build_prompt_contract(context)
        compact = build_prompt_contract(context, compact=True)
        self.assertLess(len(compact), len(standard))

    def test_structured_response_format_uses_pydantic_schema(self) -> None:
        response_format = _structured_response_format()
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "coach_summary")
        self.assertIn("properties", response_format["json_schema"]["schema"])

    def test_pydantic_summary_schema_rejects_wrong_shape(self) -> None:
        with self.assertRaises(Exception):
            CoachSummaryPayloadModel.model_validate(
                {
                    "summary": "OK",
                    "what_went_well": "not-a-list",
                }
            )

    def test_huggingface_cloud_provider_parses_valid_json(self) -> None:
        class FakeCompletions:
            calls: list[dict[str, object]] = []

            @staticmethod
            def create(**kwargs):
                FakeCompletions.calls.append(kwargs)
                assert kwargs["model"] == "Qwen/Qwen2.5-3B-Instruct"
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content=(
                                    '{"summary":"OK","what_went_well":["A"],"main_findings":["B"],'
                                    '"variation_explanation":"C","top_fixes":["D"],'
                                    '"next_session_plan":["E"],"confidence_notes":["F"]}'
                                )
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeInferenceClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.chat = FakeChat()

        fake_hf = types.SimpleNamespace(InferenceClient=FakeInferenceClient)

        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch.dict(
                "os.environ",
                {
                    "POZIFY_SUMMARY_PROVIDER": "slm_cloud",
                    "POZIFY_SUMMARY_CLOUD_MODEL": "Qwen/Qwen2.5-3B-Instruct",
                    "HF_TOKEN": "test-token",
                },
                clear=False,
            ):
                result = HuggingFaceCloudSummaryProvider().generate(
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
        self.assertEqual(result.provider, "slm_cloud")
        self.assertEqual(result.backend, "huggingface")
        self.assertIn("response_format", FakeCompletions.calls[0])

    def test_huggingface_cloud_provider_reports_parse_failure(self) -> None:
        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                del kwargs
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="not json at all"))]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeInferenceClient:
            def __init__(self, **kwargs) -> None:
                self.chat = FakeChat()

        fake_hf = types.SimpleNamespace(InferenceClient=FakeInferenceClient)

        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch.dict(
                "os.environ",
                {
                    "POZIFY_SUMMARY_PROVIDER": "slm_cloud",
                    "POZIFY_SUMMARY_CLOUD_MODEL": "Qwen/Qwen2.5-3B-Instruct",
                    "HF_TOKEN": "test-token",
                },
                clear=False,
            ):
                result = HuggingFaceCloudSummaryProvider().generate(
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

    def test_huggingface_cloud_provider_requires_token(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POZIFY_SUMMARY_PROVIDER": "slm_cloud",
                "POZIFY_SUMMARY_CLOUD_MODEL": "Qwen/Qwen2.5-3B-Instruct",
            },
            clear=True,
        ):
            result = HuggingFaceCloudSummaryProvider().generate(
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
        self.assertIn("HF_TOKEN", result.parse_error or "")

    def test_huggingface_cloud_provider_retries_without_response_format(self) -> None:
        class FakeCompletions:
            calls: list[dict[str, object]] = []

            @staticmethod
            def create(**kwargs):
                FakeCompletions.calls.append(kwargs)
                if len(FakeCompletions.calls) == 1:
                    raise RuntimeError("response_format is not supported for this model")
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content=(
                                    '{"summary":"OK","what_went_well":["A"],"main_findings":["B"],'
                                    '"variation_explanation":"C","top_fixes":["D"],'
                                    '"next_session_plan":["E"],"confidence_notes":["F"]}'
                                )
                            )
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeInferenceClient:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.chat = FakeChat()

        fake_hf = types.SimpleNamespace(InferenceClient=FakeInferenceClient)

        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch.dict(
                "os.environ",
                {
                    "POZIFY_SUMMARY_PROVIDER": "slm_cloud",
                    "POZIFY_SUMMARY_CLOUD_MODEL": "Qwen/Qwen2.5-3B-Instruct",
                    "HF_TOKEN": "test-token",
                },
                clear=False,
            ):
                result = HuggingFaceCloudSummaryProvider().generate(
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
        self.assertEqual(len(FakeCompletions.calls), 2)
        self.assertIn("response_format", FakeCompletions.calls[0])
        self.assertNotIn("response_format", FakeCompletions.calls[1])


if __name__ == "__main__":
    unittest.main()
