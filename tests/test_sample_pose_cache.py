from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify import sample_pose_cache
from pozify.contracts import PoseFrame, PoseSequence, VideoManifest, to_dict


def _manifest(video_path: Path) -> VideoManifest:
    return VideoManifest(
        video_path=str(video_path),
        fps=30.0,
        duration_sec=0.067,
        total_frames=2,
        sampled_frames=2,
        width=640,
        height=480,
        codec="mp4v",
        container="mp4",
        brightness_mean=120.0,
        blur_laplacian_var=80.0,
        quality_warnings=[],
        analysis_allowed=True,
    )


def _cached_pose_sequence() -> PoseSequence:
    frames = [
        PoseFrame(
            frame_index=0,
            timestamp_sec=0.0,
            landmarks={"left_hip": {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.9}},
            world_landmarks={},
            pose_quality={
                "source": "mediapipe_pose",
                "mean_visibility": 0.9,
                "landmark_schema": "coco17",
            },
        ),
        PoseFrame(
            frame_index=1,
            timestamp_sec=0.033,
            landmarks={"left_hip": {"x": 0.42, "y": 0.52, "z": 0.0, "visibility": 0.9}},
            world_landmarks={},
            pose_quality={
                "source": "mediapipe_pose",
                "mean_visibility": 0.9,
                "landmark_schema": "coco17",
            },
        ),
    ]
    return PoseSequence(
        frames=frames,
        normalized=True,
        smoothing_method="exponential_smoothing",
        pose_valid_ratio=1.0,
    )


class SamplePoseCacheTests(unittest.TestCase):
    def test_load_returns_cached_sequence_for_known_sample_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "sample.mp4"
            video_bytes = b"sample-video"
            video_path.write_bytes(video_bytes)

            cache_path = root / "pose_sequence.json"
            cache_path.write_text(
                json.dumps(to_dict(_cached_pose_sequence())),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    sample_pose_cache.SAMPLE_VIDEO_SHA256_ENV: hashlib.sha256(
                        video_bytes
                    ).hexdigest(),
                    sample_pose_cache.SAMPLE_POSE_CACHE_PATH_ENV: str(cache_path),
                },
            ):
                cached = sample_pose_cache.load(_manifest(video_path))

        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(len(cached.frames), 2)
        self.assertEqual(cached.frames[0].pose_quality["source"], "mediapipe_pose")

    def test_load_ignores_non_sample_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "workout.mp4"
            video_bytes = b"sample-video"
            video_path.write_bytes(video_bytes)

            cache_path = root / "pose_sequence.json"
            cache_path.write_text(
                json.dumps(to_dict(_cached_pose_sequence())),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    sample_pose_cache.SAMPLE_VIDEO_SHA256_ENV: hashlib.sha256(
                        video_bytes
                    ).hexdigest(),
                    sample_pose_cache.SAMPLE_POSE_CACHE_PATH_ENV: str(cache_path),
                },
            ):
                cached = sample_pose_cache.load(_manifest(video_path))

        self.assertIsNone(cached)


if __name__ == "__main__":
    unittest.main()
