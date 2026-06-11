from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pozify.contracts import IssueMarkers, PoseFrame, PoseSequence, Rep, Reps, VideoManifest
from pozify.steps import annotated_renderer


def _frame(frame_index: int) -> PoseFrame:
    landmarks = {
        "left_shoulder": {"x": 0.35, "y": 0.3},
        "right_shoulder": {"x": 0.65, "y": 0.3},
        "left_elbow": {"x": 0.3, "y": 0.45},
        "right_elbow": {"x": 0.7, "y": 0.45},
        "left_wrist": {"x": 0.28, "y": 0.6},
        "right_wrist": {"x": 0.72, "y": 0.6},
        "left_hip": {"x": 0.42, "y": 0.55},
        "right_hip": {"x": 0.58, "y": 0.55},
        "left_knee": {"x": 0.42, "y": 0.75},
        "right_knee": {"x": 0.58, "y": 0.75},
        "left_ankle": {"x": 0.42, "y": 0.92},
        "right_ankle": {"x": 0.58, "y": 0.92},
    }
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"source": "fake_pose", "mean_visibility": 0.95},
    )


class AnnotatedRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, frame_count: int = 6) -> Path:
        path = Path(self.temp_dir.name) / "input.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
        self.assertTrue(writer.isOpened())
        for frame_index in range(frame_count):
            frame = np.full((240, 320, 3), 90 + frame_index, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_renderer_writes_annotated_video(self) -> None:
        video_path = self._write_video()
        manifest = VideoManifest(
            video_path=str(video_path),
            fps=30.0,
            duration_sec=0.2,
            total_frames=6,
            sampled_frames=6,
            width=320,
            height=240,
            codec="mp4v",
            container="mp4",
            brightness_mean=100.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )
        pose_sequence = PoseSequence(
            frames=[_frame(index) for index in range(6)],
            normalized=True,
            smoothing_method="exponential_smoothing",
            pose_valid_ratio=1.0,
        )
        reps = Reps(
            exercise="push_up",
            reps=[Rep(1, 0, 2, 5, 0.0, 0.067, 0.167)],
            partial_reps=[],
        )

        output_path = annotated_renderer.run(
            manifest,
            pose_sequence,
            reps,
            IssueMarkers(issues=[]),
            Path(self.temp_dir.name),
        )

        self.assertIsNotNone(output_path)
        self.assertTrue(Path(str(output_path)).exists())

    def test_hdr_metadata_requires_sdr_conversion(self) -> None:
        self.assertTrue(
            annotated_renderer._needs_sdr_conversion(
                {
                    "color_space": "bt2020nc",
                    "color_transfer": "arib-std-b67",
                    "color_primaries": "bt2020",
                }
            )
        )
        self.assertFalse(
            annotated_renderer._needs_sdr_conversion(
                {
                    "color_space": "bt709",
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
