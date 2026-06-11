from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol


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


@dataclass(frozen=True)
class SlmBackendResult:
    text: str
    backend: str
    model: str


class SummarySlmBackend(Protocol):
    backend_name: str
    model_name: str

    def generate_text(self, prompt: str) -> SlmBackendResult:
        raise NotImplementedError


class TransformersSummaryBackend:
    backend_name = "transformers"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "POZIFY_SUMMARY_MODEL", "Qwen/Qwen2.5-3B-Instruct"
        )
        self.max_tokens = _env_int("POZIFY_SUMMARY_MAX_TOKENS", 512)
        self.temperature = _env_float("POZIFY_SUMMARY_TEMPERATURE", 0.2)
        self._generator = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._generator is not None:
            return
        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The local SLM summary backend requires transformers. "
                "Install the optional summary dependencies before using "
                "POZIFY_SUMMARY_PROVIDER=slm_local."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._generator = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self._tokenizer,
            trust_remote_code=True,
        )

    def _prompt_text(self, prompt: str) -> str:
        self._load()
        tokenizer = self._tokenizer
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": (
                            "You produce grounded coaching summaries from structured evidence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate_text(self, prompt: str) -> SlmBackendResult:
        prompt_text = self._prompt_text(prompt)
        assert self._generator is not None
        result = self._generator(
            prompt_text,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        if not isinstance(result, list) or not result:
            raise RuntimeError("Transformers summary backend returned no generations.")
        generation = result[0]
        if not isinstance(generation, dict):
            raise RuntimeError("Transformers summary backend returned an invalid generation payload.")
        text = generation.get("generated_text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Transformers summary backend returned empty text.")
        return SlmBackendResult(
            text=text,
            backend=self.backend_name,
            model=self.model_name,
        )


def create_summary_slm_backend(name: str | None = None) -> SummarySlmBackend:
    backend_name = (name or os.getenv("POZIFY_SUMMARY_BACKEND", "transformers")).strip().lower()
    if backend_name == "transformers":
        return TransformersSummaryBackend()
    raise ValueError(f"Unknown summary SLM backend: {backend_name!r}")
