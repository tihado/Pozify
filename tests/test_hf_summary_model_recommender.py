from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.hf_summary_model_recommender import (
    _normalize_models_payload,
    recommend_models,
    score_model,
)


class HuggingFaceSummaryModelRecommenderTests(unittest.TestCase):
    def test_normalize_models_payload_accepts_direct_list(self) -> None:
        payload = [{"id": "Qwen/Qwen2.5-7B-Instruct"}]
        records = _normalize_models_payload(payload)
        self.assertEqual(records, payload)

    def test_normalize_models_payload_accepts_data_wrapper(self) -> None:
        payload = {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
        records = _normalize_models_payload(payload)
        self.assertEqual(records, payload["data"])

    def test_normalize_models_payload_accepts_models_wrapper(self) -> None:
        payload = {"models": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}
        records = _normalize_models_payload(payload)
        self.assertEqual(records, payload["models"])

    def test_score_model_accepts_chat_instruct_candidates(self) -> None:
        candidate = score_model(
            {
                "id": "Qwen/Qwen2.5-7B-Instruct",
                "provider": "hf-inference",
                "context_length": 32768,
            }
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.model_id, "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(candidate.context_length, 32768)
        self.assertGreater(candidate.score, 0)

    def test_score_model_rejects_non_chat_models(self) -> None:
        candidate = score_model(
            {
                "id": "sentence-transformers/all-MiniLM-L6-v2",
                "provider": "hf-inference",
            }
        )
        self.assertIsNone(candidate)

    def test_recommend_models_prefers_moderate_instruct_models(self) -> None:
        recommendations = recommend_models(
            [
                {
                    "id": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "provider": "together",
                    "context_length": 8192,
                },
                {
                    "id": "openai/gpt-oss-120b",
                    "provider": "sambanova",
                    "context_length": 131072,
                },
                {
                    "id": "Qwen/Qwen2.5-7B-Instruct",
                    "provider": "hf-inference",
                    "context_length": 32768,
                },
                {
                    "id": "mistralai/Mistral-7B-Instruct-v0.3",
                    "provider": "together",
                    "context_length": 32768,
                },
            ],
            limit=3,
        )
        self.assertEqual(len(recommendations), 3)
        self.assertEqual(recommendations[0].model_id, "Qwen/Qwen2.5-7B-Instruct")
        self.assertIn(
            "mistralai/Mistral-7B-Instruct-v0.3",
            [candidate.model_id for candidate in recommendations],
        )


if __name__ == "__main__":
    unittest.main()
