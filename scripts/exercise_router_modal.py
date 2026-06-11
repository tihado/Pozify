from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import modal


APP_NAME = "pozify-exercise-router"
DATASET_ID = "RickyRiccio/Real_Time_Exercise_Recognition_Dataset"
DATA_ROOT = Path("/data")
MODEL_ROOT = Path("/models")
RAW_ROOT = DATA_ROOT / "raw"
RICCIO_ROOT = RAW_ROOT / "riccio"
CUSTOM_UNKNOWN_ROOT = RAW_ROOT / "custom_unknown"
MANIFEST_PATH = DATA_ROOT / "manifests" / "router_examples.jsonl"
FEATURE_MANIFEST_PATH = DATA_ROOT / "features" / "feature_manifest.jsonl"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
ARCHIVE_SUFFIXES = {".zip"}
RICCIO_VIDEO_COLLECTIONS = (
    "final_kaggle_with_additional_video",
    "my_test_video_1",
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libgles2", "libglib2.0-0")
    .pip_install(
        "huggingface-hub>=0.24.0",
        "joblib==1.5.3",
        "mediapipe>=0.10.35",
        "numpy==1.26.4",
        "opencv-python-headless>=4.10.0",
        "scikit-learn==1.9.0",
        "scipy==1.17.1",
        "torch>=2.2.0",
    )
    .add_local_dir("src", "/root/src", copy=True)
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name("pozify-router-data", create_if_missing=True, version=2)
model_volume = modal.Volume.from_name("pozify-router-models", create_if_missing=True, version=2)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _video_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)


def _riccio_video_files(root: Path) -> list[Path]:
    files: list[Path] = []
    dataset_roots = [
        root / "Real-Time Exercise Recognition Dataset",
        root / "Real-Time Exercise Recognition Dataset" / "Real-Time Exercise Recognition Dataset",
        root,
    ]
    for collection_name in RICCIO_VIDEO_COLLECTIONS:
        for dataset_root in dataset_roots:
            collection_root = dataset_root / collection_name
            if not collection_root.exists():
                continue
            for class_dir in sorted(path for path in collection_root.iterdir() if path.is_dir()):
                files.extend(_video_files(class_dir))
    if files:
        return sorted(files)
    return _video_files(root)


def _extract_archives(root: Path) -> list[str]:
    extracted: list[str] = []
    for archive_path in sorted(root.rglob("*")):
        if archive_path.suffix.lower() not in ARCHIVE_SUFFIXES:
            continue
        target_dir = archive_path.with_suffix("")
        marker = target_dir / ".pozify_extracted"
        if marker.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        if any(target_dir.iterdir()):
            marker.write_text("ok\n", encoding="utf-8")
            continue
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(target_dir)
        marker.write_text("ok\n", encoding="utf-8")
        extracted.append(str(archive_path))
    return extracted


def _label_from_path(path: Path) -> str:
    sys.path.insert(0, "/root/src")
    from pozify.ml.exercise_router_features import normalize_router_label

    for part in reversed(path.parts):
        label = normalize_router_label(part)
        if label != "unknown":
            return label
        if any(token in part.lower() for token in ("unknown", "curl", "idle", "walk", "stretch")):
            return "unknown"
    return "unknown"


def _example_id(path: Path) -> str:
    safe = "_".join(path.relative_to(RAW_ROOT).parts)
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in safe)


@app.function(volumes={str(DATA_ROOT): data_volume}, timeout=60 * 60)
def ingest() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    RICCIO_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=RICCIO_ROOT,
        local_dir_use_symlinks=False,
    )
    extracted_archives = _extract_archives(RICCIO_ROOT)

    examples: list[dict[str, Any]] = []
    for video_path in [*_riccio_video_files(RICCIO_ROOT), *_video_files(CUSTOM_UNKNOWN_ROOT)]:
        source = "custom_unknown" if CUSTOM_UNKNOWN_ROOT in video_path.parents else "riccio"
        examples.append(
            {
                "id": _example_id(video_path),
                "video_path": str(video_path),
                "label": "unknown" if source == "custom_unknown" else _label_from_path(video_path),
                "source": source,
            }
        )

    _write_jsonl(MANIFEST_PATH, examples)
    _write_json(
        DATA_ROOT / "manifests" / "ingest_summary.json",
        {
            "dataset_id": DATASET_ID,
            "example_count": len(examples),
            "extracted_archives": extracted_archives,
            "labels": sorted({example["label"] for example in examples}),
            "custom_unknown_root": str(CUSTOM_UNKNOWN_ROOT),
        },
    )
    data_volume.commit()
    return {"example_count": len(examples), "manifest_path": str(MANIFEST_PATH)}


@app.function(volumes={str(DATA_ROOT): data_volume})
def load_feature_examples(limit: int | None = None) -> list[dict[str, Any]]:
    examples = _read_jsonl(MANIFEST_PATH)
    return examples[:limit] if limit else examples


@app.function(volumes={str(DATA_ROOT): data_volume}, timeout=30 * 60)
def extract_example_features(example: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, "/root/src")

    import numpy as np

    from pozify.ml.exercise_router_features import extract_router_windows
    from pozify.steps import pose_cleaning, pose_landmarker, video_qc

    video_path = Path(example["video_path"])
    manifest = video_qc.run(str(video_path))
    raw_sequence = pose_landmarker.run(manifest, mock=False)
    sequence = pose_cleaning.run(raw_sequence)
    windows = extract_router_windows(sequence)

    feature_path = DATA_ROOT / "features" / f"{example['id']}.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if windows:
        np.savez_compressed(
            feature_path,
            vectors=np.stack([window.vector for window in windows]),
            tensors=np.stack([window.tensor for window in windows]),
            label=example["label"],
        )
    else:
        np.savez_compressed(
            feature_path,
            vectors=np.empty((0, 0), dtype=np.float32),
            tensors=np.empty((0, 0, 0), dtype=np.float32),
            label=example["label"],
        )

    data_volume.commit()
    return {
        "id": example["id"],
        "label": example["label"],
        "video_path": str(video_path),
        "feature_path": str(feature_path),
        "window_count": len(windows),
        "pose_valid_ratio": sequence.pose_valid_ratio,
    }


@app.function(volumes={str(DATA_ROOT): data_volume})
def write_feature_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _write_jsonl(FEATURE_MANIFEST_PATH, rows)
    _write_json(
        DATA_ROOT / "features" / "feature_summary.json",
        {
            "example_count": len(rows),
            "window_count": sum(int(row.get("window_count", 0)) for row in rows),
            "failed_count": sum(1 for row in rows if not row.get("ok", True)),
        },
    )
    data_volume.commit()
    return {"feature_manifest_path": str(FEATURE_MANIFEST_PATH), "example_count": len(rows)}


def _load_feature_arrays() -> tuple[Any, Any, Any]:
    import numpy as np

    vectors: list[np.ndarray] = []
    tensors: list[np.ndarray] = []
    labels: list[str] = []
    for row in _read_jsonl(FEATURE_MANIFEST_PATH):
        if not row.get("ok", True) or int(row.get("window_count", 0)) <= 0:
            continue
        data = np.load(row["feature_path"], allow_pickle=False)
        row_vectors = data["vectors"]
        row_tensors = data["tensors"]
        label = str(data["label"])
        for index in range(row_vectors.shape[0]):
            vectors.append(row_vectors[index])
            tensors.append(row_tensors[index])
            labels.append(label)
    if not vectors:
        return None, None, []
    return np.stack(vectors), np.stack(tensors), labels


@app.function(volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume}, timeout=30 * 60)
def train_baseline() -> dict[str, Any]:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sys.path.insert(0, "/root/src")
    from pozify.ml.exercise_router_features import window_vector_feature_names

    vectors, _tensors, labels = _load_feature_arrays()
    if vectors is None or len(set(labels)) < 2:
        result = {"ok": False, "error": "At least two labels with extracted windows are required"}
        _write_json(MODEL_ROOT / "baseline_metrics.json", result)
        model_volume.commit()
        return result

    label_counts = {label: labels.count(label) for label in sorted(set(labels))}
    stratify = labels if min(label_counts.values()) >= 2 else None
    train_x, valid_x, train_y, valid_y = train_test_split(
        vectors,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)
    metrics = {
        "ok": True,
        "model_kind": "baseline",
        "accuracy": round(float(accuracy_score(valid_y, predictions)), 4),
        "label_counts": label_counts,
        "window_count": int(vectors.shape[0]),
    }
    joblib.dump(
        {
            "model": model,
            "labels": [str(label) for label in model.classes_],
            "model_kind": "baseline",
            "feature_names": window_vector_feature_names(),
            "metrics": metrics,
        },
        MODEL_ROOT / "baseline.joblib",
    )
    _write_json(MODEL_ROOT / "baseline_metrics.json", metrics)
    model_volume.commit()
    return metrics


@app.function(gpu="T4", volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume}, timeout=60 * 60)
def train_temporal(epochs: int = 8) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, "/root/src")
    from pozify.ml.exercise_router_features import ROUTER_LABELS

    _vectors, tensors, labels = _load_feature_arrays()
    if tensors is None or len(set(labels)) < 2:
        result = {"ok": False, "error": "At least two labels with extracted windows are required"}
        _write_json(MODEL_ROOT / "temporal_metrics.json", result)
        model_volume.commit()
        return result

    label_to_index = {label: index for index, label in enumerate(ROUTER_LABELS)}
    x = torch.tensor(tensors, dtype=torch.float32)
    y = torch.tensor([label_to_index.get(label, label_to_index["unknown"]) for label in labels])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    class TinyGRU(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=64, batch_first=True)
            self.head = nn.Linear(64, len(ROUTER_LABELS))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            _outputs, hidden = self.gru(inputs)
            return self.head(hidden[-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGRU(input_size=int(x.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    last_loss = 0.0
    for _epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    torch.save(
        {
            "model_kind": "temporal",
            "architecture": "tiny_gru",
            "state_dict": model.cpu().state_dict(),
            "labels": list(ROUTER_LABELS),
            "input_size": int(x.shape[-1]),
        },
        MODEL_ROOT / "temporal.pt",
    )
    metrics = {
        "ok": True,
        "model_kind": "temporal",
        "architecture": "tiny_gru",
        "epochs": epochs,
        "final_training_loss": round(last_loss, 4),
        "window_count": int(x.shape[0]),
    }
    _write_json(MODEL_ROOT / "temporal_metrics.json", metrics)
    model_volume.commit()
    return metrics


@app.function(volumes={str(DATA_ROOT): data_volume, str(MODEL_ROOT): model_volume})
def evaluate() -> dict[str, Any]:
    import joblib

    sys.path.insert(0, "/root/src")
    from pozify.ml.exercise_router_evaluation import evaluation_to_dict, evaluate_router_predictions

    vectors, _tensors, labels = _load_feature_arrays()
    baseline_path = MODEL_ROOT / "baseline.joblib"
    if vectors is None or not baseline_path.exists():
        result = {"ok": False, "error": "Feature arrays and baseline.joblib are required"}
        _write_json(MODEL_ROOT / "evaluation.json", result)
        model_volume.commit()
        return result

    artifact = joblib.load(baseline_path)
    predictions = list(artifact["model"].predict(vectors))
    evaluation = evaluation_to_dict(evaluate_router_predictions(labels, predictions))
    result = {
        "ok": True,
        "selected_model": "baseline.joblib",
        **evaluation,
    }
    _write_json(MODEL_ROOT / "evaluation.json", result)
    shutil.copyfile(baseline_path, MODEL_ROOT / "router.joblib")
    model_volume.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "evaluate", limit: int | None = None, epochs: int = 8) -> None:
    if stage == "ingest":
        print(ingest.remote())
    elif stage == "features":
        examples = load_feature_examples.remote(limit)
        rows: list[dict[str, Any]] = []
        for result in extract_example_features.map(examples, return_exceptions=True):
            if isinstance(result, Exception):
                rows.append({"ok": False, "error": repr(result)})
            else:
                rows.append({"ok": True, **result})
        print(write_feature_manifest.remote(rows))
    elif stage == "train-baseline":
        print(train_baseline.remote())
    elif stage == "train-temporal":
        print(train_temporal.remote(epochs=epochs))
    elif stage == "evaluate":
        print(evaluate.remote())
    elif stage == "all":
        print(ingest.remote())
        examples = load_feature_examples.remote(limit)
        rows = []
        for result in extract_example_features.map(examples, return_exceptions=True):
            rows.append({"ok": False, "error": repr(result)} if isinstance(result, Exception) else {"ok": True, **result})
        print(write_feature_manifest.remote(rows))
        print(train_baseline.remote())
        print(train_temporal.remote(epochs=epochs))
        print(evaluate.remote())
    else:
        raise ValueError(
            "stage must be one of: ingest, features, train-baseline, train-temporal, evaluate, all"
        )
