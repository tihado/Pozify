from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import request


MODELS_ENDPOINT = "https://router.huggingface.co/v1/models"
DEFAULT_LIMIT = 3
CHAT_HINTS = ("instruct", "chat", "assistant")
PREFERRED_FAMILIES = (
    "qwen",
    "llama",
    "mistral",
    "deepseek",
    "gpt-oss",
    "glm",
)
AVOID_HINTS = (
    "vision",
    "vl",
    "audio",
    "asr",
    "tts",
    "embedding",
    "rerank",
    "image",
    "diffusion",
    "whisper",
)


@dataclass(frozen=True)
class RecommendedModel:
    model_id: str
    provider_hint: str | None
    context_length: int | None
    score: float
    reasons: list[str]


def _auth_token(explicit_token: str | None = None) -> str:
    token = explicit_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN is required.")
    return token


def fetch_router_models(token: str | None = None) -> list[dict[str, Any]]:
    req = request.Request(
        MODELS_ENDPOINT,
        headers={
            "Authorization": f"Bearer {_auth_token(token)}",
            "Accept": "application/json",
        },
    )
    with request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _normalize_models_payload(payload)


def _normalize_models_payload(payload: Any) -> list[dict[str, Any]]:
    records: Any = payload
    if isinstance(payload, dict):
        for key in ("data", "models", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
    if not isinstance(records, list):
        raise RuntimeError(
            "Unexpected /v1/models payload; expected a list or an object with "
            "'data'/'models' list entries."
        )
    return [item for item in records if isinstance(item, dict)]


def _first_str(values: list[Any]) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_model_id(record: dict[str, Any]) -> str | None:
    return _first_str([
        record.get("id"),
        record.get("model"),
        record.get("name"),
    ])


def _extract_provider_hint(record: dict[str, Any]) -> str | None:
    provider = _first_str([record.get("provider"), record.get("inference_provider")])
    if provider:
        return provider
    providers = record.get("providers")
    if isinstance(providers, list) and providers:
        for provider_record in providers:
            if isinstance(provider_record, dict):
                provider_name = _first_str(
                    [
                        provider_record.get("provider"),
                        provider_record.get("name"),
                        provider_record.get("id"),
                    ]
                )
                if provider_name:
                    return provider_name
            elif isinstance(provider_record, str) and provider_record.strip():
                return provider_record.strip()
    return None


def _extract_context_length(record: dict[str, Any]) -> int | None:
    candidates = [
        record.get("context_length"),
        record.get("max_input_tokens"),
        record.get("max_context_length"),
        record.get("contextWindow"),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
        if isinstance(candidate, float) and candidate.is_integer() and candidate > 0:
            return int(candidate)
    providers = record.get("providers")
    if isinstance(providers, list):
        for provider_record in providers:
            if not isinstance(provider_record, dict):
                continue
            nested = _extract_context_length(provider_record)
            if nested:
                return nested
    return None


def _looks_chat_capable(model_id: str) -> bool:
    lowered = model_id.lower()
    if any(hint in lowered for hint in AVOID_HINTS):
        return False
    return any(hint in lowered for hint in CHAT_HINTS)


def _size_penalty(model_id: str) -> float:
    lowered = model_id.lower()
    penalties = {
        "405b": 2.0,
        "120b": 1.7,
        "70b": 1.2,
        "34b": 0.6,
        "32b": 0.5,
        "27b": 0.4,
        "14b": 0.1,
        "13b": 0.1,
    }
    for marker, penalty in penalties.items():
        if marker in lowered:
            return penalty
    return 0.0


def score_model(record: dict[str, Any]) -> RecommendedModel | None:
    model_id = _extract_model_id(record)
    if model_id is None:
        return None

    lowered = model_id.lower()
    if not _looks_chat_capable(model_id):
        return None

    reasons: list[str] = []
    score = 0.0

    for family in PREFERRED_FAMILIES:
        if family in lowered:
            score += 2.0
            reasons.append(f"preferred family: {family}")
            break

    if "instruct" in lowered:
        score += 2.0
        reasons.append("instruction-tuned")
    if "chat" in lowered:
        score += 1.5
        reasons.append("chat-tuned")

    context_length = _extract_context_length(record)
    if context_length is not None:
        if context_length >= 32000:
            score += 1.5
            reasons.append(">=32k context")
        elif context_length >= 16000:
            score += 1.0
            reasons.append(">=16k context")
        elif context_length >= 8000:
            score += 0.6
            reasons.append(">=8k context")

    provider_hint = _extract_provider_hint(record)
    if provider_hint:
        score += 0.4
        reasons.append(f"provider hint: {provider_hint}")

    penalty = _size_penalty(model_id)
    score -= penalty
    if penalty:
        reasons.append("size penalty for large model")

    return RecommendedModel(
        model_id=model_id,
        provider_hint=provider_hint,
        context_length=context_length,
        score=round(score, 2),
        reasons=reasons,
    )


def recommend_models(
    records: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[RecommendedModel]:
    ranked = [candidate for record in records if (candidate := score_model(record)) is not None]
    ranked.sort(
        key=lambda candidate: (
            -candidate.score,
            -(candidate.context_length or 0),
            candidate.model_id,
        )
    )
    return ranked[:limit]
