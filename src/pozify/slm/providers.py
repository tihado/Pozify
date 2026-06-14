from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

from pozify.env import env_truthy, load_local_env
from pozify.hf_spaces import default_spaces_gpu_duration, spaces_gpu


PROVIDER_ENV = "POZIFY_COACH_SUMMARY_PROVIDER"
MODEL_ENV = "POZIFY_COACH_SUMMARY_MODEL"
DISABLE_REMOTE_ENV = "POZIFY_COACH_SUMMARY_DISABLE_REMOTE"
MAX_TOKENS_ENV = "POZIFY_COACH_SUMMARY_MAX_TOKENS"
TEMPERATURE_ENV = "POZIFY_COACH_SUMMARY_TEMPERATURE"
HF_TOKEN_ENV = "HF_TOKEN"

DEFAULT_PROVIDER = "hf_inference"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LOCAL_TRANSFORMERS_PROVIDER = "local_transformers"
LOCAL_TRANSFORMERS_ALIASES = {LOCAL_TRANSFORMERS_PROVIDER, "local", "transformers"}

_LOCAL_TRANSFORMERS_CACHE: dict[str, tuple[Any, Any]] = {}


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


def _gpu_duration(*_args: object, **_kwargs: object) -> int:
    return default_spaces_gpu_duration()


def _load_local_transformers_backend(model: str, token: str | None) -> tuple[Any, Any]:
    cache_key = f"{model}:{token or ''}"
    cached = _LOCAL_TRANSFORMERS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency declared in project
        raise RuntimeError(
            "transformers and torch are required for local coach summary inference"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)
    language_model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map="auto",
        torch_dtype="auto",
        token=token,
    )
    language_model.eval()

    cache_entry = (tokenizer, language_model)
    _LOCAL_TRANSFORMERS_CACHE[cache_key] = cache_entry
    return cache_entry


def _local_transformers_device(language_model: Any) -> Any:
    device = getattr(language_model, "device", None)
    if device is not None:
        return device

    try:
        return next(language_model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Local coach summary model has no parameters") from exc


@spaces_gpu(duration=_gpu_duration)
def _generate_local_transformers_summary(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    token: str | None,
) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency declared in project
        raise RuntimeError("torch is required for local coach summary inference") from exc

    tokenizer, language_model = _load_local_transformers_backend(model, token)
    messages = [
        {
            "role": "system",
            "content": "Return JSON only.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)

    inputs = tokenizer([text], return_tensors="pt")
    model_inputs = inputs.to(_local_transformers_device(language_model))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = language_model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[-1]
    generated = generated[:, prompt_length:]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Local transformers inference returned empty output")
    return text


class LocalTransformersCoachSummaryModel:
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

    def generate_summary(self, prompt: str) -> CoachSummaryGeneration:
        text = _generate_local_transformers_summary(
            model=self.model,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            token=self.token,
        )
        return CoachSummaryGeneration(
            text=text,
            provider=LOCAL_TRANSFORMERS_PROVIDER,
            model=self.model,
        )


def get_coach_summary_model() -> CoachSummaryModel | None:
    load_local_env()

    provider = os.getenv(PROVIDER_ENV, DEFAULT_PROVIDER).strip().lower()
    if provider == "hf_inference":
        if env_truthy(os.getenv(DISABLE_REMOTE_ENV)):
            return None
        return HFInferenceCoachSummaryModel(
            model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            temperature=_env_float(TEMPERATURE_ENV, 0.1),
            token=os.getenv(HF_TOKEN_ENV),
        )

    if provider in LOCAL_TRANSFORMERS_ALIASES:
        return LocalTransformersCoachSummaryModel(
            model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
            max_tokens=_env_int(MAX_TOKENS_ENV, 700),
            temperature=_env_float(TEMPERATURE_ENV, 0.1),
            token=os.getenv(HF_TOKEN_ENV),
        )

    if env_truthy(os.getenv(DISABLE_REMOTE_ENV)):
        return None

    return None
