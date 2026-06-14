from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import modal


APP_NAME = "pozify-coach-summary"
DEFAULT_HF_REPO_ID = "build-small-hackathon/pozify-coach-summary"
HF_REPO_ID_ENV = "POZIFY_COACH_SUMMARY_HF_REPO_ID"
HF_PRIVATE_ENV = "POZIFY_COACH_SUMMARY_HF_PRIVATE"
DATA_ROOT = Path("/data")
MODEL_ROOT = Path("/models")
ROOT_DATA = Path("/root/data")
ROOT_CONFIGS = Path("/root/configs")
SFT_ROOT = DATA_ROOT / "sft"
MODEL_CARD_PATH = MODEL_ROOT / "README.md"
DEFAULT_CONFIG_PATH = ROOT_CONFIGS / "coach_summary_lora.default.json"
TRAINING_CONFIG_PATH = MODEL_ROOT / "training_config.json"
TRAINING_SUMMARY_PATH = MODEL_ROOT / "training_summary.json"
EVALUATION_PATH = MODEL_ROOT / "evaluation.json"
HF_UPLOAD_PATH = MODEL_ROOT / "hf_upload.json"
DEFAULT_ADAPTER_DIR = MODEL_ROOT / "adapter"
HF_METADATA_FILENAMES = (
    "training_config.json",
    "training_summary.json",
    "evaluation.json",
    "hf_upload.json",
)
HF_DATA_FILENAMES = (
    "coach_summary_train.jsonl",
    "coach_summary_eval.jsonl",
    "public_fitness_style.jsonl",
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "accelerate>=0.34.0",
        "bitsandbytes>=0.43.1",
        "datasets>=2.20.0",
        "huggingface-hub>=0.24.0",
        "peft>=0.12.0",
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "trl>=0.10.1",
    )
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_dir("data", "/root/data", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(
    "pozify-coach-summary-data", create_if_missing=True, version=2
)
model_volume = modal.Volume.from_name(
    "pozify-coach-summary-models", create_if_missing=True, version=2
)


def _load_local_env_vars(filename: str = ".env") -> dict[str, str]:
    candidates = (
        Path.cwd() / filename,
        Path(__file__).resolve().parents[1] / filename,
    )
    values: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            values[key] = value.strip().strip("'").strip('"')
    return values


def _hf_secret() -> modal.Secret:
    env_values = _load_local_env_vars()
    secret_payload: dict[str, str] = {}
    for key in ("HF_TOKEN", HF_REPO_ID_ENV, HF_PRIVATE_ENV):
        value = os.getenv(key, env_values.get(key))
        if value is not None and str(value).strip():
            secret_payload[key] = str(value).strip()
    return modal.Secret.from_dict(secret_payload)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains a non-object JSONL row")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _load_config() -> dict[str, Any]:
    config = _read_json(DEFAULT_CONFIG_PATH)
    if TRAINING_CONFIG_PATH.exists():
        config.update(_read_json(TRAINING_CONFIG_PATH))
    return config


def _render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"].capitalize()
        parts.append(f"{role}: {message['content'].strip()}")
    return "\n\n".join(parts)


def _sample_style_rows(
    *,
    style_rows: list[dict[str, Any]],
    train_count: int,
    style_weight: float,
) -> list[dict[str, Any]]:
    if not style_rows or style_weight <= 0:
        return []
    keep_count = min(len(style_rows), int(round(train_count * style_weight)))
    return style_rows[:keep_count]


def _build_training_dataset_rows(
    *,
    train_rows: list[dict[str, Any]],
    style_rows: list[dict[str, Any]],
    style_weight: float,
) -> list[dict[str, str]]:
    selected_style_rows = _sample_style_rows(
        style_rows=style_rows,
        train_count=len(train_rows),
        style_weight=style_weight,
    )
    merged = [*train_rows, *selected_style_rows]
    return [{"text": _render_messages(row["messages"])} for row in merged]


def _build_eval_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"text": _render_messages(row["messages"])} for row in rows]


def _model_card_text(
    *,
    repo_id: str,
    config: dict[str, Any],
    training_summary: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "LoRA adapter for Pozify grounded coach-summary generation.",
        "",
        "## Base Model",
        "",
        f"- `{config.get('base_model', 'unknown')}`",
        "",
        "## Training Data",
        "",
        f"- Train file: `{config.get('train_file')}`",
        f"- Eval file: `{config.get('eval_file')}`",
        f"- Style file: `{config.get('style_file', SFT_ROOT / 'public_fitness_style.jsonl')}`",
        "",
        "## Objective",
        "",
        "Generate grounded `coach_summary.json` outputs from Pozify structured evidence and knowledge cards.",
        "",
    ]
    if training_summary:
        lines.extend(
            [
                "## Training Summary",
                "",
                f"- Train rows: `{training_summary.get('train_row_count')}`",
                f"- Eval rows: `{training_summary.get('eval_row_count')}`",
                f"- Style rows mixed in: `{training_summary.get('style_row_count')}`",
                f"- Output dir: `{training_summary.get('output_dir')}`",
                "",
            ]
        )
    if evaluation:
        lines.extend(
            [
                "## Evaluation",
                "",
                f"- JSON validity rate: `{evaluation.get('json_validity_rate')}`",
                f"- Verifier pass rate: `{evaluation.get('verifier_pass_rate')}`",
                f"- Section completeness rate: `{evaluation.get('section_completeness_rate')}`",
                "",
            ]
        )
    return "\n".join(lines)


def _verifier_inputs_from_evidence(payload: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    sys.path.insert(0, "/root/src")
    from pozify.contracts import (
        ExerciseClassification,
        IssueMarker,
        IssueMarkers,
        Rep,
        RepAnalysis,
        RepAnalysisItem,
        Reps,
        Variation,
    )

    classification_payload = payload["exercise_classification"]
    variation_payload = payload["variation"]
    rep_summary_payload = payload["rep_summary"]
    issue_summary_payload = payload["issue_summary"]

    classification = ExerciseClassification(
        exercise=str(classification_payload["exercise"]),
        confidence=float(classification_payload["confidence"]),
        window_predictions=[],
        fallback_required=bool(classification_payload.get("fallback_required", False)),
    )
    variation = Variation(
        exercise=str(variation_payload["exercise"]),
        detected_variation=str(variation_payload["detected_variation"]),
        variation_confidence=float(variation_payload["variation_confidence"]),
        not_issues=[str(item) for item in variation_payload.get("not_issues", [])],
    )
    rep_metrics = rep_summary_payload.get("rep_metrics", [])
    reps = Reps(
        exercise=str(classification.exercise),
        reps=[
            Rep(
                rep_id=int(item.get("rep_id", index + 1)),
                start_frame=0,
                mid_frame=0,
                end_frame=0,
                start_sec=0.0,
                mid_sec=0.0,
                end_sec=float(item.get("duration_sec", 0.0)),
            )
            for index, item in enumerate(rep_metrics)
        ],
        partial_reps=[],
    )
    analysis = RepAnalysis(
        exercise=str(classification.exercise),
        items=[
            RepAnalysisItem(
                rep_id=int(item.get("rep_id", index + 1)),
                duration_sec=float(item.get("duration_sec", 0.0)),
                range_of_motion_score=float(item.get("range_of_motion_score", 0.0)),
                stability_score=float(item.get("stability_score", 0.0)),
                symmetry_score=float(item.get("symmetry_score", 0.0)),
                metrics=dict(item.get("metrics", {})),
                variation_hints=[str(value) for value in item.get("variation_hints", [])],
            )
            for index, item in enumerate(rep_metrics)
        ],
        aggregate_metrics=dict(rep_summary_payload.get("aggregate_metrics", {})),
    )
    issues = IssueMarkers(
        issues=[
            IssueMarker(
                rep_id=int(item.get("rep_id", 0)),
                issue=str(item["issue"]),
                severity=float(item.get("severity", 0.0)),
                start_frame=int(item.get("start_frame", 0)),
                end_frame=int(item.get("end_frame", 0)),
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 0.0)),
                affected_joints=[str(value) for value in item.get("affected_joints", [])],
                evidence=dict(item.get("evidence", {})),
            )
            for item in issue_summary_payload.get("issues", [])
        ]
    )
    return classification, variation, reps, analysis, issues


def _generate_json_only(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@app.function(
    volumes={str(DATA_ROOT): data_volume},
    timeout=20 * 60,
)
def prepare_data() -> dict[str, Any]:
    SFT_ROOT.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    counts: dict[str, int] = {}
    for filename in HF_DATA_FILENAMES:
        source = ROOT_DATA / "sft" / filename
        if not source.exists():
            continue
        target = SFT_ROOT / filename
        shutil.copyfile(source, target)
        copied.append(filename)
        counts[filename] = len(_read_jsonl(target))

    summary = {
        "ok": bool(copied),
        "copied_files": copied,
        "row_counts": counts,
        "sft_root": str(SFT_ROOT),
    }
    _write_json(DATA_ROOT / "prepare_data_summary.json", summary)
    data_volume.commit()
    return summary


@app.function(
    gpu="A10G",
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    secrets=[_hf_secret()],
    timeout=3 * 60 * 60,
)
def train(
    epochs: int | None = None,
    style_weight: float = 0.2,
    output_subdir: str = "adapter",
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    config = _load_config()
    if epochs is not None:
        config["num_train_epochs"] = epochs
    config["style_weight"] = style_weight
    config["style_file"] = str(SFT_ROOT / "public_fitness_style.jsonl")

    train_rows = _read_jsonl(SFT_ROOT / "coach_summary_train.jsonl")
    eval_rows = _read_jsonl(SFT_ROOT / "coach_summary_eval.jsonl")
    style_rows = _read_jsonl(SFT_ROOT / "public_fitness_style.jsonl")
    if not train_rows or not eval_rows:
        result = {
            "ok": False,
            "error": "Missing SFT train/eval rows. Run prepare_data first.",
        }
        _write_json(TRAINING_SUMMARY_PATH, result)
        model_volume.commit()
        return result

    training_rows = _build_training_dataset_rows(
        train_rows=train_rows,
        style_rows=style_rows,
        style_weight=style_weight,
    )
    eval_dataset_rows = _build_eval_dataset_rows(eval_rows)
    train_dataset = Dataset.from_list(training_rows)
    eval_dataset = Dataset.from_list(eval_dataset_rows)

    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        quantization_config=quantization_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    adapter_dir = MODEL_ROOT / output_subdir
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=int(config["max_seq_length"]),
        args=SFTConfig(
            output_dir=str(adapter_dir),
            learning_rate=float(config["learning_rate"]),
            num_train_epochs=float(config["num_train_epochs"]),
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_steps=10,
            bf16=True,
            report_to=[],
        ),
    )
    train_result = trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    summary = {
        "ok": True,
        "base_model": config["base_model"],
        "adapter_dir": str(adapter_dir),
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "style_row_count": len(training_rows) - len(train_rows),
        "merged_train_row_count": len(training_rows),
        "style_weight": style_weight,
        "epochs": config["num_train_epochs"],
        "global_step": int(getattr(train_result, "global_step", 0)),
        "training_loss": float(getattr(train_result, "training_loss", 0.0)),
        "output_dir": str(adapter_dir),
    }
    _write_json(TRAINING_CONFIG_PATH, config)
    _write_json(TRAINING_SUMMARY_PATH, summary)
    model_volume.commit()
    return summary


@app.function(
    gpu="A10G",
    volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume},
    secrets=[_hf_secret()],
    timeout=90 * 60,
)
def evaluate(
    adapter_subdir: str = "adapter",
    limit: int | None = None,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    sys.path.insert(0, "/root/src")
    from pozify.steps import verifier
    from pozify.steps.coach_summary import _extract_json_object, _summary_from_payload

    config = _load_config()
    adapter_dir = MODEL_ROOT / adapter_subdir
    eval_rows = _read_jsonl(SFT_ROOT / "coach_summary_eval.jsonl")
    if limit is not None:
        eval_rows = eval_rows[:limit]
    if not adapter_dir.exists():
        result = {"ok": False, "error": f"Adapter dir not found: {adapter_dir}"}
        _write_json(EVALUATION_PATH, result)
        model_volume.commit()
        return result

    tokenizer = AutoTokenizer.from_pretrained(str(config["base_model"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    json_valid_count = 0
    verifier_pass_count = 0
    section_complete_count = 0
    evaluated = 0
    failures: list[dict[str, Any]] = []
    required_sections = {
        "summary",
        "what_you_did",
        "what_looked_good",
        "what_changed_across_reps",
        "valid_variation_vs_issue",
        "top_fixes",
        "next_session_plan",
        "confidence_notes",
    }

    for index, row in enumerate(eval_rows):
        evaluated += 1
        prompt = _render_messages(row["messages"][:2])
        try:
            generated_text = _generate_json_only(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=int(config.get("max_new_tokens", 700)),
            )
            payload = _extract_json_object(generated_text)
            json_valid_count += 1
            if required_sections <= payload.keys():
                section_complete_count += 1
            summary = _summary_from_payload(payload)
            evidence_payload = json.loads(row["messages"][1]["content"])
            classification, variation, reps, analysis, issues = _verifier_inputs_from_evidence(
                evidence_payload
            )
            verification = verifier.run(
                summary,
                issues,
                variation,
                classification=classification,
                analysis=analysis,
                reps=reps,
            )
            if verification.passed:
                verifier_pass_count += 1
            else:
                failures.append(
                    {
                        "index": index,
                        "reason": "verifier_failed",
                        "notes": verification.notes,
                    }
                )
        except Exception as exc:  # pragma: no cover - runtime failure path
            failures.append({"index": index, "reason": repr(exc)})

    result = {
        "ok": True,
        "adapter_dir": str(adapter_dir),
        "evaluated_count": evaluated,
        "json_valid_count": json_valid_count,
        "json_validity_rate": round(json_valid_count / evaluated, 4) if evaluated else 0.0,
        "verifier_pass_count": verifier_pass_count,
        "verifier_pass_rate": round(verifier_pass_count / evaluated, 4) if evaluated else 0.0,
        "section_completeness_rate": round(section_complete_count / evaluated, 4)
        if evaluated
        else 0.0,
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    _write_json(EVALUATION_PATH, result)
    model_volume.commit()
    return result


def _upload_hf_file(
    api: Any,
    *,
    repo_id: str,
    local_path: Path,
    path_in_repo: str,
) -> dict[str, Any]:
    if not local_path.exists():
        return {
            "path": str(local_path),
            "path_in_repo": path_in_repo,
            "uploaded": False,
            "reason": "missing",
        }
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
    )
    return {
        "path": str(local_path),
        "path_in_repo": path_in_repo,
        "uploaded": True,
    }


@app.function(
    volumes={str(MODEL_ROOT): model_volume, str(DATA_ROOT): data_volume},
    secrets=[_hf_secret()],
    timeout=30 * 60,
)
def publish_to_hf(
    repo_id: str | None = None,
    private: bool | None = None,
    adapter_subdir: str = "adapter",
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    repo_id = repo_id or os.getenv(HF_REPO_ID_ENV) or DEFAULT_HF_REPO_ID
    private = _env_truthy(os.getenv(HF_PRIVATE_ENV)) if private is None else private
    if not os.getenv("HF_TOKEN"):
        return {
            "ok": False,
            "error": "HF_TOKEN is required in the Modal environment or local .env",
            "repo_id": repo_id,
        }

    config = _read_json(TRAINING_CONFIG_PATH) if TRAINING_CONFIG_PATH.exists() else {}
    training_summary = _read_json(TRAINING_SUMMARY_PATH) if TRAINING_SUMMARY_PATH.exists() else None
    evaluation = _read_json(EVALUATION_PATH) if EVALUATION_PATH.exists() else None
    MODEL_CARD_PATH.write_text(
        _model_card_text(
            repo_id=repo_id,
            config=config,
            training_summary=training_summary,
            evaluation=evaluation,
        ),
        encoding="utf-8",
    )

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    uploads = [
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_CARD_PATH,
            path_in_repo="README.md",
        )
    ]
    adapter_dir = MODEL_ROOT / adapter_subdir
    if adapter_dir.exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            path_in_repo="adapter",
        )
        uploads.append(
            {
                "path": str(adapter_dir),
                "path_in_repo": "adapter/",
                "uploaded": True,
            }
        )
    else:
        uploads.append(
            {
                "path": str(adapter_dir),
                "path_in_repo": "adapter/",
                "uploaded": False,
                "reason": "missing",
            }
        )
    uploads.extend(
        _upload_hf_file(
            api,
            repo_id=repo_id,
            local_path=MODEL_ROOT / filename,
            path_in_repo=filename,
        )
        for filename in HF_METADATA_FILENAMES
    )

    result = {
        "ok": any(item["uploaded"] for item in uploads),
        "repo_id": repo_id,
        "private": private,
        "uploads": uploads,
    }
    _write_json(HF_UPLOAD_PATH, result)
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(
    stage: str = "evaluate",
    epochs: int | None = None,
    style_weight: float = 0.2,
    limit: int | None = None,
    repo_id: str | None = None,
    private: bool | None = None,
) -> None:
    if stage == "prepare-data":
        print(prepare_data.remote())
    elif stage == "train":
        print(train.remote(epochs=epochs, style_weight=style_weight))
    elif stage == "evaluate":
        print(evaluate.remote(limit=limit))
    elif stage == "publish":
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    elif stage == "all":
        print(prepare_data.remote())
        print(train.remote(epochs=epochs, style_weight=style_weight))
        print(evaluate.remote(limit=limit))
        print(publish_to_hf.remote(repo_id=repo_id, private=private))
    else:
        raise ValueError(
            "stage must be one of: prepare-data, train, evaluate, publish, all"
        )
