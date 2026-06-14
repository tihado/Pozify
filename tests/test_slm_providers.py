from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.slm.providers import (  # noqa: E402
    get_coach_summary_model,
    HFInferenceCoachSummaryModel,
    LocalTransformersCoachSummaryModel,
)


class SlmProviderTests(unittest.TestCase):
    def test_returns_local_transformers_model_when_local_dir_is_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POZIFY_COACH_SUMMARY_LOCAL_MODEL_DIR": "/tmp/local-model",
                "POZIFY_COACH_SUMMARY_BASE_MODEL": "Qwen/Qwen3-14B",
                "POZIFY_COACH_SUMMARY_ADAPTER_ID": "pozify/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, LocalTransformersCoachSummaryModel)

    def test_returns_hf_inference_model_when_remote_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POZIFY_COACH_SUMMARY_MODEL": "Qwen/Qwen3-14B",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)

    def test_remote_model_uses_runtime_model_not_adapter_repo(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POZIFY_COACH_SUMMARY_MODEL": "build-small-hackathon/pozify-coach-summary",
                "POZIFY_COACH_SUMMARY_ADAPTER_ID": "pozify/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)
        self.assertEqual(model.model, "build-small-hackathon/pozify-coach-summary")

    def test_remote_model_falls_back_to_default_when_only_adapter_repo_is_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POZIFY_COACH_SUMMARY_ADAPTER_ID": "pozify/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)
        self.assertEqual(model.model, "build-small-hackathon/pozify-coach-summary1")

    def test_hf_inference_falls_back_to_text_generation_for_non_chat_model(self) -> None:
        class _TextGenerationClient:
            def __init__(self) -> None:
                self.text_generation_kwargs = None

            def chat_completion(self, **_kwargs):
                raise RuntimeError("not a chat model")

            def text_generation(self, prompt: str, **kwargs):
                self.text_generation_kwargs = {"prompt": prompt, **kwargs}
                return '{"summary":"ok"}'

        client = _TextGenerationClient()
        model = HFInferenceCoachSummaryModel(
            model="build-small-hackathon/pozify-coach-summary1",
            max_tokens=123,
            temperature=0.2,
        )
        model._client = client

        generation = model.generate_summary("coach prompt")

        self.assertEqual(generation.provider, "hf_inference")
        self.assertEqual(generation.model, "build-small-hackathon/pozify-coach-summary1")
        self.assertEqual(generation.text, '{"summary":"ok"}')
        self.assertEqual(
            client.text_generation_kwargs,
            {
                "prompt": "coach prompt",
                "model": "build-small-hackathon/pozify-coach-summary1",
                "max_new_tokens": 123,
                "temperature": 0.2,
                "return_full_text": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
