from __future__ import annotations

from pozify.contracts import PoseFrame, PoseSequence


def run(sequence: PoseSequence) -> PoseSequence:
    cleaned_frames: list[PoseFrame] = []
    for frame in sequence.frames:
        cleaned_frames.append(
            PoseFrame(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                landmarks=frame.landmarks,
                world_landmarks=frame.world_landmarks,
                pose_quality={
                    **frame.pose_quality,
                    "cleaned": True,
                    "normalization_origin": "mid_hip_mock",
                    "normalization_scale": "torso_length_mock",
                },
            )
        )

    return PoseSequence(
        frames=cleaned_frames,
        normalized=True,
        smoothing_method="mock_exponential_smoothing",
        pose_valid_ratio=sequence.pose_valid_ratio,
    )

