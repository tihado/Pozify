from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
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
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}
HDR_PRIMARIES = {"bt2020"}
BT709_COLOR_ARGS = (
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
)


def _tool_path(name: str) -> str | None:
    return shutil.which(name)


def _video_color_metadata(video_path: str) -> dict[str, str]:
    ffprobe = _tool_path("ffprobe")
    if ffprobe is None:
        return {}

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries,color_range",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return {}
    stream = streams[0]
    if not isinstance(stream, dict):
        return {}
    metadata: dict[str, str] = {}
    for key, value in stream.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        normalized_value = str(value).lower()
        if normalized_value == "unknown":
            continue
        metadata[key] = normalized_value
    return metadata


def _needs_sdr_conversion(color_metadata: dict[str, str]) -> bool:
    transfer = color_metadata.get("color_transfer", "")
    primaries = color_metadata.get("color_primaries", "")
    return transfer in HDR_TRANSFERS or primaries in HDR_PRIMARIES


def _sdr_filter(color_metadata: dict[str, str]) -> str:
    transfer = color_metadata.get("color_transfer", "arib-std-b67")
    primaries = color_metadata.get("color_primaries", "bt2020")
    matrix = color_metadata.get("color_space", "bt2020nc")
    if transfer not in HDR_TRANSFERS:
        transfer = "arib-std-b67"
    if primaries not in HDR_PRIMARIES:
        primaries = "bt2020"
    if matrix not in {"bt2020nc", "bt2020c"}:
        matrix = "bt2020nc"

    return (
        f"zscale=transfer=linear:transferin={transfer}:"
        f"primariesin={primaries}:matrixin={matrix}:npl=100,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=transfer=bt709:primaries=bt709:matrix=bt709:range=tv,"
        "format=yuv420p"
    )


def _transcode_hdr_to_sdr(
    input_path: Path,
    output_path: Path,
    color_metadata: dict[str, str],
) -> bool:
    ffmpeg = _tool_path("ffmpeg")
    if ffmpeg is None:
        return False

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-vf",
        _sdr_filter(color_metadata),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        *BT709_COLOR_ARGS,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _encode_bt709_output(
    raw_video_path: Path,
    output_path: Path,
    audio_source_path: Path | None,
) -> bool:
    ffmpeg = _tool_path("ffmpeg")
    if ffmpeg is None:
        return False

    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(raw_video_path),
    ]
    if audio_source_path is not None:
        command.extend(["-i", str(audio_source_path)])

    command.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if audio_source_path is not None:
        command.extend(["-map", "1:a?"])

    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-vf",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709,format=yuv420p",
            "-pix_fmt",
            "yuv420p",
            *BT709_COLOR_ARGS,
        ]
    )
    if audio_source_path is not None:
        command.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
    else:
        command.append("-an")
    command.append(str(output_path))

    try:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        return False
    return output_path.exists() and output_path.stat().st_size > 0


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


def _open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> tuple[cv2.VideoWriter | None, str | None]:
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

    source_path = Path(manifest.video_path)
    color_metadata = _video_color_metadata(manifest.video_path)
    render_input_path = source_path
    temporary_paths: list[Path] = []
    if _needs_sdr_conversion(color_metadata):
        sdr_input_path = run_dir / "renderer_sdr_input.mp4"
        if _transcode_hdr_to_sdr(source_path, sdr_input_path, color_metadata):
            render_input_path = sdr_input_path
            temporary_paths.append(sdr_input_path)

    capture = cv2.VideoCapture(str(render_input_path))
    if not capture.isOpened():
        capture.release()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
        return manifest.video_path

    fps = manifest.fps if manifest.fps > 0 else 30.0
    width = manifest.width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = manifest.height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return manifest.video_path

    output_path = run_dir / "annotated_video.mp4"
    raw_output_path = run_dir / "annotated_video_raw.mp4" if _tool_path("ffmpeg") else output_path
    writer, _codec = _open_video_writer(raw_output_path, fps, width, height)
    if writer is None:
        capture.release()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
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
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    if raw_output_path != output_path:
        encoded = _encode_bt709_output(raw_output_path, output_path, source_path)
        if not encoded:
            raw_output_path.replace(output_path)
        raw_output_path.unlink(missing_ok=True)

    return str(output_path)
