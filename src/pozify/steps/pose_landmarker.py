from __future__ import annotations

from pozify.contracts import PoseFrame, PoseSequence, VideoManifest
from pozify.steps.video_qc import sample_frame_indices


LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _mock_landmarks(frame_index: int) -> dict[str, dict[str, float]]:
    phase = (frame_index % 90) / 90
    vertical_offset = 0.08 if phase < 0.5 else 0.0
    landmarks: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        side_offset = -0.06 if "left" in name else 0.06 if "right" in name else 0.0
        base_y = 0.2 + min(idx, 11) * 0.045
        landmarks[name] = {
            "x": round(0.5 + side_offset, 4),
            "y": round(base_y + vertical_offset, 4),
            "z": round(-0.02 * side_offset, 4),
            "visibility": 0.95,
        }
    return landmarks


def run(manifest: VideoManifest) -> PoseSequence:
    frames: list[PoseFrame] = []
    for frame_index in sample_frame_indices(manifest.total_frames):
        frames.append(
            PoseFrame(
                frame_index=frame_index,
                timestamp_sec=round(frame_index / manifest.fps, 3),
                landmarks=_mock_landmarks(frame_index),
                world_landmarks={},
                pose_quality={
                    "mean_visibility": 0.95,
                    "critical_landmarks_visible": True,
                    "mock": True,
                },
            )
        )

    return PoseSequence(
        frames=frames,
        normalized=False,
        smoothing_method="none",
        pose_valid_ratio=1.0,
    )
