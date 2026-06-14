from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pozify.contracts import (
    ContractValidationError,
    PoseFrame,
    PoseSequence,
    VideoManifest,
    validate_contract,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_POSE_CACHE_PATH_ENV = "POZIFY_SAMPLE_POSE_CACHE_PATH"
SAMPLE_POSE_RUNS_DIR_ENV = "POZIFY_SAMPLE_POSE_RUNS_DIR"
SAMPLE_VIDEO_SHA256_ENV = "POZIFY_SAMPLE_VIDEO_SHA256"

SAMPLE_VIDEO_NAMES = {
    "sample.mp4",
    "sample.mov",
    "sample_converted.mp4",
}
KNOWN_SAMPLE_VIDEO_SHA256 = {
    "77e17441c273da24647028f20b5573d03710e0f9b32dc43ff1dc3cc72aaa77f4",
}
KNOWN_SAMPLE_VIDEO_SIZES = {
    9_797_579,
}


def _env_values(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [value.strip() for value in raw_value.split(os.pathsep) if value.strip()]


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _known_sample_hashes() -> set[str]:
    hashes = set(KNOWN_SAMPLE_VIDEO_SHA256)
    hashes.update(_env_values(SAMPLE_VIDEO_SHA256_ENV))

    fixture_path = ROOT_DIR / "tests" / "fixtures" / "sample.mp4"
    if fixture_path.is_file():
        fixture_hash = _sha256(fixture_path)
        if fixture_hash is not None:
            hashes.add(fixture_hash)

    return hashes


def _looks_like_sample_name(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    return (
        name in SAMPLE_VIDEO_NAMES
        or stem == "sample"
        or stem.startswith("pozify-sample-")
    )


def _is_known_sample_video(manifest: VideoManifest) -> bool:
    if not manifest.analysis_allowed or manifest.video_path is None:
        return False

    video_path = Path(manifest.video_path)
    if not video_path.is_file():
        return False

    try:
        file_size = video_path.stat().st_size
    except OSError:
        return False

    should_hash = _looks_like_sample_name(video_path) or file_size in KNOWN_SAMPLE_VIDEO_SIZES
    if not should_hash:
        return False

    video_hash = _sha256(video_path)
    return video_hash in _known_sample_hashes() if video_hash is not None else False


def _manifest_matches(manifest: VideoManifest, payload: dict[str, Any]) -> bool:
    try:
        return (
            int(payload["total_frames"]) == manifest.total_frames
            and int(payload["width"]) == manifest.width
            and int(payload["height"]) == manifest.height
            and round(float(payload["fps"]), 3) == round(manifest.fps, 3)
            and round(float(payload["duration_sec"]), 3)
            == round(manifest.duration_sec, 3)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _manifest_path_looks_like_sample(payload: dict[str, Any]) -> bool:
    video_path = payload.get("video_path")
    return isinstance(video_path, str) and _looks_like_sample_name(Path(video_path))


def _runs_cache_paths(manifest: VideoManifest) -> list[Path]:
    runs_dir_values = _env_values(SAMPLE_POSE_RUNS_DIR_ENV)
    runs_dir = Path(runs_dir_values[0]) if runs_dir_values else ROOT_DIR / "runs"
    if not runs_dir.is_dir():
        return []

    candidates: list[Path] = []
    for manifest_path in runs_dir.glob("*/video_manifest.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue
        if not _manifest_path_looks_like_sample(payload):
            continue
        if not _manifest_matches(manifest, payload):
            continue

        pose_path = manifest_path.parent / "pose_sequence.json"
        if pose_path.is_file():
            candidates.append(pose_path)

    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _cache_paths(manifest: VideoManifest) -> list[Path]:
    paths = [Path(value) for value in _env_values(SAMPLE_POSE_CACHE_PATH_ENV)]

    default_cache = ROOT_DIR / "demo" / "precomputed" / "sample_pose_sequence.json"
    if default_cache.is_file():
        paths.append(default_cache)

    paths.extend(_runs_cache_paths(manifest))
    return paths


def _pose_payload_matches_manifest(payload: dict[str, Any], manifest: VideoManifest) -> bool:
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return False
    if manifest.total_frames > 0 and len(frames) != manifest.total_frames:
        return False
    if payload.get("normalized") is not True:
        return False

    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        return False
    first_quality = first_frame.get("pose_quality", {})
    if not isinstance(first_quality, dict):
        return False
    if first_quality.get("source") in {"mock_pose", "none"}:
        return False
    if first_quality.get("landmark_schema") != "coco17":
        return False

    last_frame = frames[-1]
    if not isinstance(last_frame, dict):
        return False
    try:
        return int(last_frame["frame_index"]) <= max(manifest.total_frames - 1, 0)
    except (KeyError, TypeError, ValueError):
        return False


def _pose_sequence_from_payload(payload: dict[str, Any]) -> PoseSequence:
    return PoseSequence(
        frames=[
            PoseFrame(
                frame_index=int(frame["frame_index"]),
                timestamp_sec=float(frame["timestamp_sec"]),
                landmarks=frame["landmarks"],
                world_landmarks=frame["world_landmarks"],
                pose_quality=frame["pose_quality"],
            )
            for frame in payload["frames"]
        ],
        normalized=bool(payload["normalized"]),
        smoothing_method=str(payload["smoothing_method"]),
        pose_valid_ratio=float(payload["pose_valid_ratio"]),
    )


def _load_pose_sequence(path: Path, manifest: VideoManifest) -> PoseSequence | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    try:
        validate_contract("pose_sequence.json", payload)
    except ContractValidationError:
        return None

    if not _pose_payload_matches_manifest(payload, manifest):
        return None

    return _pose_sequence_from_payload(payload)


def load(manifest: VideoManifest) -> PoseSequence | None:
    if not _is_known_sample_video(manifest):
        return None

    for cache_path in _cache_paths(manifest):
        cached_sequence = _load_pose_sequence(cache_path, manifest)
        if cached_sequence is not None:
            return cached_sequence

    return None
