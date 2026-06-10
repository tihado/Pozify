from __future__ import annotations

from pathlib import Path

from pozify.contracts import IssueMarkers, VideoManifest


def run(manifest: VideoManifest, issues: IssueMarkers, run_dir: Path) -> str | None:
    placeholder = {
        "status": "mock_renderer",
        "message": "Annotated renderer is not implemented yet. Returning original video path.",
        "issue_count": len(issues.issues),
    }
    (run_dir / "annotated_video_placeholder.json").write_text(
        __import__("json").dumps(placeholder, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest.video_path

