from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
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
        self.device = os.getenv("POZIFY_SUMMARY_DEVICE", "cpu").strip().lower()
        self.max_tokens = _env_int("POZIFY_SUMMARY_MAX_TOKENS", 512)
        self.temperature = _env_float("POZIFY_SUMMARY_TEMPERATURE", 0.2)
        self._generator = None
        self._tokenizer = None

    def _pipeline_device(self) -> int | str | None:
        if self.device == "cpu":
            return -1
        if self.device == "mps":
            return "mps"
        if self.device == "cuda":
            return 0
        if self.device == "auto":
            return None
        raise RuntimeError(
            "Unsupported POZIFY_SUMMARY_DEVICE value. Use one of: cpu, mps, cuda, auto."
        )

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
        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model_name,
            "tokenizer": self._tokenizer,
            "trust_remote_code": True,
        }
        device = self._pipeline_device()
        if device is not None:
            pipeline_kwargs["device"] = device
        self._generator = pipeline(**pipeline_kwargs)

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


class LlamaCppGgufSummaryBackend:
    backend_name = "gguf"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "POZIFY_SUMMARY_MODEL", "bartowski/Qwen2.5-3B-Instruct-GGUF"
        )
        self.filename = os.getenv(
            "POZIFY_SUMMARY_GGUF_FILENAME", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        )
        self.local_path = os.getenv("POZIFY_SUMMARY_GGUF_PATH")
        self.max_tokens = _env_int("POZIFY_SUMMARY_MAX_TOKENS", 512)
        self.temperature = _env_float("POZIFY_SUMMARY_TEMPERATURE", 0.2)
        self.context_window = _env_int("POZIFY_SUMMARY_CONTEXT_WINDOW", 4096)
        self.threads = _env_int("POZIFY_SUMMARY_THREADS", 0)
        self.gpu_layers = _env_int("POZIFY_SUMMARY_GPU_LAYERS", 0)
        self._llm = None

    def _resolve_model_path(self) -> str:
        if self.local_path:
            local_path = Path(self.local_path).expanduser()
            if not local_path.is_file():
                raise RuntimeError(
                    f"Configured POZIFY_SUMMARY_GGUF_PATH does not exist: {local_path}"
                )
            return str(local_path)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "The GGUF summary backend requires huggingface_hub to download the model file."
            ) from exc

        try:
            return hf_hub_download(repo_id=self.model_name, filename=self.filename)
        except Exception as exc:  # pragma: no cover - exact exception depends on hub runtime
            raise RuntimeError(
                "Failed to download the configured GGUF summary model. "
                "Check POZIFY_SUMMARY_MODEL, POZIFY_SUMMARY_GGUF_FILENAME, and HF_TOKEN if needed."
            ) from exc

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "The GGUF summary backend requires llama-cpp-python. "
                "Install the optional summary dependencies before using "
                "POZIFY_SUMMARY_BACKEND=gguf."
            ) from exc

        model_path = self._resolve_model_path()
        load_kwargs = {
            "model_path": model_path,
            "n_ctx": self.context_window,
            "verbose": False,
        }
        if self.threads > 0:
            load_kwargs["n_threads"] = self.threads
        if self.gpu_layers >= 0:
            load_kwargs["n_gpu_layers"] = self.gpu_layers
        self._llm = Llama(**load_kwargs)

    def generate_text(self, prompt: str) -> SlmBackendResult:
        self._load()
        assert self._llm is not None
        result = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You produce grounded coaching summaries from structured evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("GGUF summary backend returned no choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("GGUF summary backend returned an invalid choice payload.")
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("GGUF summary backend returned no message payload.")
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("GGUF summary backend returned empty text.")
        return SlmBackendResult(
            text=text,
            backend=self.backend_name,
            model=f"{self.model_name}:{self.filename}",
        )


def create_summary_slm_backend(name: str | None = None) -> SummarySlmBackend:
    backend_name = (name or os.getenv("POZIFY_SUMMARY_BACKEND", "transformers")).strip().lower()
    if backend_name == "transformers":
        return TransformersSummaryBackend()
    if backend_name == "gguf":
        return LlamaCppGgufSummaryBackend()
    raise ValueError(f"Unknown summary SLM backend: {backend_name!r}")
