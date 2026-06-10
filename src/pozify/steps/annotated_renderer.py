from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from pozify.contracts import IssueMarkers, PoseSequence, Reps, VideoManifest


SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

PREFERRED_VIDEO_CODECS = ("avc1", "H264", "mp4v")


def _frame_landmark_points(
    frame_landmarks: dict[str, dict[str, float]],
    width: int,
    height: int,
) -> dict[str, tuple[int, int]]:
    points: dict[str, tuple[int, int]] = {}
    for name, values in frame_landmarks.items():
        x = values.get("x")
        y = values.get("y")
        if x is None or y is None:
            continue
        points[name] = (int(round(x * width)), int(round(y * height)))
    return points


def _draw_pose(frame: Any, points: dict[str, tuple[int, int]]) -> None:
    for start_name, end_name in SKELETON_EDGES:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, (90, 220, 90), 2)

    for point in points.values():
        cv2.circle(frame, point, 3, (255, 240, 40), -1)


def _rep_boundaries(reps: Reps) -> dict[int, list[str]]:
    boundaries: dict[int, list[str]] = {}
    for rep in reps.reps:
        boundaries.setdefault(rep.start_frame, []).append(f"rep {rep.rep_id} start")
        boundaries.setdefault(rep.mid_frame, []).append(f"rep {rep.rep_id} mid")
        boundaries.setdefault(rep.end_frame, []).append(f"rep {rep.rep_id} end")
    return boundaries


def _draw_overlays(
    frame: Any,
    frame_index: int,
    rep_count: int,
    issue_count: int,
    boundary_labels: dict[int, list[str]],
) -> None:
    cv2.putText(
        frame,
        f"Reps detected: {rep_count}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Issues: {issue_count}",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 230, 255),
        2,
        cv2.LINE_AA,
    )
    labels = boundary_labels.get(frame_index, [])
    for offset, label in enumerate(labels):
        cv2.putText(
            frame,
            label,
            (16, 96 + offset * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )


def _open_video_writer(output_path: Path, fps: float, width: int, height: int) -> tuple[cv2.VideoWriter | None, str | None]:
    for codec in PREFERRED_VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def run(
    manifest: VideoManifest,
    pose_sequence: PoseSequence,
    reps: Reps,
    issues: IssueMarkers,
    run_dir: Path,
) -> str | None:
    if not manifest.analysis_allowed or not manifest.video_path:
        return manifest.video_path

    capture = cv2.VideoCapture(manifest.video_path)
    if not capture.isOpened():
        capture.release()
        return manifest.video_path

    fps = manifest.fps if manifest.fps > 0 else 30.0
    width = manifest.width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = manifest.height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return manifest.video_path

    output_path = run_dir / "annotated_video.mp4"
    writer, _codec = _open_video_writer(output_path, fps, width, height)
    if writer is None:
        capture.release()
        return manifest.video_path

    pose_by_frame = {frame.frame_index: frame for frame in pose_sequence.frames}
    ordered_pose_frames = sorted(pose_sequence.frames, key=lambda frame: frame.frame_index)
    pose_cursor = 0
    last_pose_frame = None
    boundary_labels = _rep_boundaries(reps)

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if pose_cursor < len(ordered_pose_frames):
                while (
                    pose_cursor + 1 < len(ordered_pose_frames)
                    and ordered_pose_frames[pose_cursor + 1].frame_index <= frame_index
                ):
                    pose_cursor += 1
                candidate = ordered_pose_frames[pose_cursor]
                if candidate.frame_index <= frame_index:
                    last_pose_frame = candidate

            exact_pose = pose_by_frame.get(frame_index)
            active_pose = exact_pose or last_pose_frame
            if active_pose is not None and active_pose.landmarks:
                points = _frame_landmark_points(active_pose.landmarks, width, height)
                _draw_pose(frame, points)

            _draw_overlays(frame, frame_index, len(reps.reps), len(issues.issues), boundary_labels)
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()

    return str(output_path)
