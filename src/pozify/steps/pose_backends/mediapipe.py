from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from pozify.hf_spaces import zero_gpu_enabled
from pozify.steps.pose_backends.base import PoseDetection
from pozify.steps.pose_backends.landmarks import landmark_list_to_dict


POSE_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


class MediaPipePoseBackend:
    source = "mediapipe_pose"

    def __init__(self) -> None:
        self._pose = self._create_pose()

    def __enter__(self) -> "MediaPipePoseBackend":
        enter = getattr(self._pose, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *args: object) -> None:
        exit_method = getattr(self._pose, "__exit__", None)
        if exit_method is not None:
            exit_method(*args)
            return
        close = getattr(self._pose, "close", None)
        if close is not None:
            close()

    def detect(self, rgb_frame: Any | None, *, frame_index: int) -> PoseDetection:
        if rgb_frame is None:
            return PoseDetection(landmarks={}, world_landmarks={}, source=self.source)
        results = self._pose.process(rgb_frame)
        return PoseDetection(
            landmarks=landmark_list_to_dict(getattr(results, "pose_landmarks", None)),
            world_landmarks=landmark_list_to_dict(
                getattr(results, "pose_world_landmarks", None)
            ),
            source=self.source,
        )

    def _create_pose(self) -> Any:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for the mediapipe pose backend. Install dependencies with uv sync."
            ) from exc

        if hasattr(mp, "solutions"):
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=False,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        return _MediaPipeTasksPoseAdapter(mp, _ensure_pose_task_model())


class _MediaPipeTasksPoseAdapter:
    def __init__(self, mediapipe_module: Any, model_path: Path) -> None:
        self._mp = mediapipe_module
        self._landmarker = self._create_landmarker(model_path)

    def _create_landmarker(self, model_path: Path) -> Any:
        delegate = self._preferred_delegate()
        try:
            return self._create_landmarker_with_delegate(model_path, delegate)
        except Exception:
            cpu_delegate = self._cpu_delegate()
            if delegate == cpu_delegate:
                raise
            return self._create_landmarker_with_delegate(model_path, cpu_delegate)

    def _create_landmarker_with_delegate(self, model_path: Path, delegate: Any) -> Any:
        base_options = self._mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=delegate,
        )
        options = self._mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return self._mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def _preferred_delegate(self) -> Any:
        if not zero_gpu_enabled():
            return self._cpu_delegate()

        delegate = self._mp.tasks.BaseOptions.Delegate
        return getattr(delegate, "GPU", self._cpu_delegate())

    def _cpu_delegate(self) -> Any:
        return self._mp.tasks.BaseOptions.Delegate.CPU

    def process(self, rgb_frame: Any) -> Any:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        pose_world_landmarks = (
            result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        )
        return type(
            "PoseResult",
            (),
            {
                "pose_landmarks": pose_landmarks,
                "pose_world_landmarks": pose_world_landmarks,
            },
        )()

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if close is not None:
            close()


def _pose_task_model_path() -> Path:
    configured_path = os.getenv("POZIFY_MEDIAPIPE_POSE_MODEL")
    if configured_path:
        return Path(configured_path).expanduser()

    cache_path = Path(os.getenv("POZIFY_MODEL_CACHE", "/tmp/pozify-models"))
    return cache_path / "pose_landmarker_lite.task"


def _ensure_pose_task_model() -> Path:
    model_path = _pose_task_model_path()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(POSE_TASK_MODEL_URL, model_path)
    return model_path
