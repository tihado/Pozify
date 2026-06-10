from __future__ import annotations

# ruff: noqa: E402

import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import PoseFrame, PoseSequence, VideoManifest
from pozify.steps import pose_cleaning, pose_landmarker
from pozify.steps.pose_backends import PoseDetection, landmark_list_to_dict


def _landmark(
    x: float, y: float, z: float = 0.0, visibility: float = 0.9
) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _landmark_result(offset: float = 0.0) -> SimpleNamespace:
    landmarks = [
        _landmark(0.2 + offset + index * 0.001, 0.1 + index * 0.01, -0.01, 0.91)
        for index in range(33)
    ]
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks),
        pose_world_landmarks=SimpleNamespace(landmark=landmarks),
    )


class _FakePose:
    source = "fake_pose"

    def __init__(self) -> None:
        self.calls = 0

    def __enter__(self) -> "_FakePose":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def detect(self, _frame: object, *, frame_index: int) -> PoseDetection:
        self.calls += 1
        result = _landmark_result(offset=self.calls * 0.01)
        return PoseDetection(
            landmarks=landmark_list_to_dict(result.pose_landmarks),
            world_landmarks=landmark_list_to_dict(result.pose_world_landmarks),
            source=self.source,
        )


class PoseStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, frame_count: int = 4) -> Path:
        path = Path(self.temp_dir.name) / "pose.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
        )
        self.assertTrue(writer.isOpened())
        for frame_index in range(frame_count):
            frame = np.full((480, 640, 3), 120 + frame_index, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_pose_landmarker_keeps_all_33_landmarks_and_quality(self) -> None:
        path = self._write_video()
        manifest = VideoManifest(
            video_path=str(path),
            fps=30.0,
            duration_sec=0.133,
            total_frames=4,
            sampled_frames=4,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )
        sequence = pose_landmarker.run(manifest, backend=_FakePose())

        self.assertEqual(len(sequence.frames), 4)
        self.assertEqual(len(sequence.frames[0].landmarks), 33)
        self.assertIn("left_foot_index", sequence.frames[0].landmarks)
        self.assertEqual(len(sequence.frames[0].world_landmarks), 33)
        self.assertEqual(sequence.frames[0].pose_quality["source"], "fake_pose")
        self.assertGreater(sequence.frames[0].pose_quality["mean_visibility"], 0.9)
        self.assertTrue(sequence.frames[0].pose_quality["critical_landmarks_visible"])
        self.assertEqual(sequence.pose_valid_ratio, 1.0)

    def test_pose_landmarker_uses_dense_frames_for_real_backend(self) -> None:
        path = self._write_video(frame_count=130)
        manifest = VideoManifest(
            video_path=str(path),
            fps=30.0,
            duration_sec=4.333,
            total_frames=130,
            sampled_frames=12,
            width=640,
            height=480,
            codec="mp4v",
            container="mp4",
            brightness_mean=120.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )

        sequence = pose_landmarker.run(manifest, backend=_FakePose())

        self.assertEqual(len(sequence.frames), 130)

    def test_pose_cleaning_interpolates_smooths_and_adds_normalized_fields(
        self,
    ) -> None:
        first_landmarks = _landmark_result(offset=0.0).pose_landmarks
        last_landmarks = _landmark_result(offset=0.2).pose_landmarks
        first = landmark_list_to_dict(first_landmarks)
        last = landmark_list_to_dict(last_landmarks)
        sequence = PoseSequence(
            frames=[
                PoseFrame(0, 0.0, first, {}, {"mean_visibility": 0.9}),
                PoseFrame(1, 0.033, {}, {}, {"mean_visibility": 0.0}),
                PoseFrame(2, 0.067, last, {}, {"mean_visibility": 0.9}),
            ],
            normalized=False,
            smoothing_method="none",
            pose_valid_ratio=0.667,
        )

        cleaned = pose_cleaning.run(sequence)

        self.assertTrue(cleaned.normalized)
        self.assertEqual(cleaned.smoothing_method, "exponential_smoothing")
        self.assertEqual(cleaned.pose_valid_ratio, 1.0)
        self.assertTrue(cleaned.frames[1].pose_quality["interpolated"])
        shoulder = cleaned.frames[1].landmarks["left_shoulder"]
        self.assertIn("x", shoulder)
        self.assertIn("smoothed_x", shoulder)
        self.assertIn("normalized_x", shoulder)
        self.assertTrue(cleaned.frames[1].pose_quality["normalized"])

    @unittest.skipUnless(
        os.getenv("POZIFY_RUN_REAL_POSE_TESTS") == "1",
        "set POZIFY_RUN_REAL_POSE_TESTS=1 to run the real MediaPipe fixture smoke test",
    )
    def test_real_sample_mov_extracts_pose_landmarks(self) -> None:
        path = FIXTURES_DIR / "sample.MOV"
        self.assertTrue(path.exists(), path)

        from pozify.steps import video_qc

        manifest = video_qc.run(str(path))
        sequence = pose_landmarker.run(manifest, mock=False, backend_name="mediapipe")
        cleaned = pose_cleaning.run(sequence)

        self.assertTrue(manifest.analysis_allowed)
        self.assertGreater(len(sequence.frames), 0)
        self.assertGreater(sequence.pose_valid_ratio, 0.0)
        self.assertEqual(len(sequence.frames[0].landmarks), 33)
        self.assertEqual(sequence.frames[0].pose_quality["source"], "mediapipe_pose")
        self.assertTrue(cleaned.normalized)


if __name__ == "__main__":
    unittest.main()
