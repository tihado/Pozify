from __future__ import annotations

from typing import Any


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def landmark_to_dict(landmark: Any) -> dict[str, float]:
    payload = {
        "x": round(float(landmark.x), 6),
        "y": round(float(landmark.y), 6),
        "z": round(float(landmark.z), 6),
        "visibility": round(float(getattr(landmark, "visibility", 0.0)), 6),
    }
    presence = getattr(landmark, "presence", None)
    if presence is not None:
        payload["presence"] = round(float(presence), 6)
    return payload


def landmark_list_to_dict(landmarks: Any | None) -> dict[str, dict[str, float]]:
    if landmarks is None:
        return {}
    landmark_values = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    return {
        name: landmark_to_dict(landmark)
        for name, landmark in zip(LANDMARK_NAMES, landmark_values, strict=False)
    }
