from __future__ import annotations

from pozify.steps.pose_backends.base import PoseBackend
from pozify.steps.pose_backends.mediapipe import MediaPipePoseBackend
from pozify.steps.pose_backends.mmpose import MMPoseBackend
from pozify.steps.pose_backends.mock import MockPoseBackend


def create_pose_backend(name: str) -> PoseBackend:
    normalized_name = name.strip().lower().replace("-", "_")
    if normalized_name == "mock":
        return MockPoseBackend()
    if normalized_name == "mediapipe":
        return MediaPipePoseBackend()
    if normalized_name == "mmpose":
        return MMPoseBackend()
    raise ValueError(f"Unknown pose backend: {name!r}")
