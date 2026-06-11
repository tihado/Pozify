from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

from pozify.env import env_truthy, load_local_env


PROVIDER_ENV = "POZIFY_COACH_SUMMARY_PROVIDER"
MODEL_ENV = "POZIFY_COACH_SUMMARY_MODEL"
DISABLE_REMOTE_ENV = "POZIFY_COACH_SUMMARY_DISABLE_REMOTE"
MAX_TOKENS_ENV = "POZIFY_COACH_SUMMARY_MAX_TOKENS"
TEMPERATURE_ENV = "POZIFY_COACH_SUMMARY_TEMPERATURE"
HF_TOKEN_ENV = "HF_TOKEN"

DEFAULT_PROVIDER = "hf_inference"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class CoachSummaryGeneration:
    text: str
    provider: str
    model: str


class CoachSummaryModel(Protocol):
    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        ...


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class HFInferenceCoachSummaryModel:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 700,
        temperature: float = 0.1,
        token: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.token = token
        self._client = None

    def _client_instance(self):
        if self._client is not None:
            return self._client
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:  # pragma: no cover - dependency declared in project
            raise RuntimeError("huggingface_hub is required for Hugging Face inference") from exc
        self._client = InferenceClient(api_key=self.token)
        return self._client

    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        client = self._client_instance()
        if hasattr(client, "chat_completion"):
            response = client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("Hugging Face inference returned no choices")
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", None)
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Hugging Face inference returned an empty message")
            return CoachSummaryGeneration(
                text=text,
                provider="hf_inference",
                model=self.model,
            )

        if hasattr(client, "text_generation"):
            text = client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                return_full_text=False,
            )
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Hugging Face text generation returned empty output")
            return CoachSummaryGeneration(
                text=text,
                provider="hf_inference",
                model=self.model,
            )

        raise RuntimeError("No supported Hugging Face inference method is available")


def get_coach_summary_model() -> CoachSummaryModel | None:
    load_local_env()

    if env_truthy(os.getenv(DISABLE_REMOTE_ENV)):
        return None

    provider = os.getenv(PROVIDER_ENV, DEFAULT_PROVIDER).strip().lower()
    if provider != "hf_inference":
        return None

    return HFInferenceCoachSummaryModel(
        model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
        max_tokens=_env_int(MAX_TOKENS_ENV, 700),
        temperature=_env_float(TEMPERATURE_ENV, 0.1),
        token=os.getenv(HF_TOKEN_ENV),
    )
