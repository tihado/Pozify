from __future__ import annotations

from pathlib import Path

from pozify.contracts import VideoManifest


def run(video_path: str | None) -> VideoManifest:
    warnings: list[str] = []
    if video_path is None:
        warnings.append("no_video_uploaded_mock_mode")
    elif not Path(video_path).exists():
        warnings.append("video_path_not_found_mock_mode")

    return VideoManifest(
        video_path=video_path,
        fps=30.0,
        duration_sec=12.0,
        total_frames=360,
        sampled_frames=180,
        quality_warnings=warnings,
        analysis_allowed=True,
    )

