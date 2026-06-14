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
                "POZIFY_COACH_SUMMARY_BASE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
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
                "POZIFY_COACH_SUMMARY_MODEL": "Qwen/Qwen2.5-7B-Instruct",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)

    def test_remote_model_prefers_adapter_repo_when_present(self) -> None:
        with patch.dict(
            os.environ,
            {
                "POZIFY_COACH_SUMMARY_ADAPTER_ID": "pozify/coach-summary-lora",
            },
            clear=True,
        ):
            model = get_coach_summary_model()

        self.assertIsInstance(model, HFInferenceCoachSummaryModel)
        self.assertEqual(model.model, "pozify/coach-summary-lora")


if __name__ == "__main__":
    unittest.main()
