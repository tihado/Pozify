from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pozify.exercise_catalog import USER_SELECTABLE_EXERCISES
from pozify.pipeline import run_pipeline

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
RUNS_ROOT = BASE_DIR / "runs"

APP_DESCRIPTION = (
    "Upload a short workout clip, tune the athlete context, and generate an annotated "
    "form-review report with structured artifacts."
)


QUALITY_GUIDANCE = {
    "too_short": "Record at least 10 seconds so the set contains enough movement context.",
    "too_long": "Keep the clip under 60 seconds for the MVP analyzer.",
    "too_dark": "Use brighter, even lighting and keep the body visible against the background.",
    "too_blurry": "Stabilize the camera and avoid fast panning or heavy motion blur.",
    "fps_too_low": "Use a camera mode with at least 15 FPS.",
    "resolution_too_low": "Record at 480x360 or higher so joint positions are readable.",
    "video_decode_failed": "Upload a playable video file; the current file could not be decoded.",
}


def _pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _quality_markdown(video_manifest: dict[str, Any]) -> str:
    warnings = video_manifest["quality_warnings"]
    if not warnings:
        return "## Video Quality\n\nNo quality warnings detected."

    warning_items = "\n".join(f"- `{warning}`: {QUALITY_GUIDANCE[warning]}" for warning in warnings)
    status = (
        "Analysis is blocked until the video can be decoded reliably."
        if not video_manifest["analysis_allowed"]
        else "Analysis completed, but capture quality may affect downstream feedback."
    )
    return f"""## Capture Quality

{status}

{warning_items}
"""


def _mock_status_markdown(report: dict[str, Any]) -> str:
    mock_steps = report["artifacts"].get("mock_steps", [])
    if not mock_steps:
        return ""
    steps = ", ".join(f"`{step}`" for step in mock_steps)
    return (
        "## Pipeline Status\n\n"
        "The current run uses real video QC, pose extraction, rep segmentation, "
        "rep analysis, variation detection, and annotated video rendering, "
        f"but these steps still use placeholders: {steps}."
    )


def _metrics_markdown(report: dict[str, Any]) -> str:
    metrics = report["rep_analysis"]["aggregate_metrics"]
    lines = [
        "## Movement Metrics",
        "",
        f"- **Average ROM score:** {metrics.get('avg_rom_score', 0):.0%}",
        f"- **Average stability score:** {metrics.get('avg_stability_score', 0):.0%}",
        f"- **Average symmetry score:** {metrics.get('avg_symmetry_score', 0):.0%}",
        f"- **Average rep duration:** {metrics.get('avg_rep_duration_sec', 0)}s",
        f"- **Tempo consistency:** {metrics.get('avg_tempo_consistency_score', 0):.0%}",
        f"- **ROM fatigue trend:** {metrics.get('fatigue_trend_rom_delta', 0):+.2f}",
    ]
    if "avg_hand_width_ratio" in metrics:
        lines.append(f"- **Hand width ratio:** {metrics['avg_hand_width_ratio']:.2f}")
    if "avg_stance_width_ratio" in metrics:
        lines.append(f"- **Stance width ratio:** {metrics['avg_stance_width_ratio']:.2f}")
    if "avg_lockout_quality" in metrics:
        lines.append(f"- **Lockout quality:** {metrics['avg_lockout_quality']:.0%}")
    return "\n".join(lines)


def analyze_video(
    video_path: str | None,
    goal: str,
    experience_level: str,
    intended_exercise: str,
    intended_variation: str,
    limitations: list[str],
    equipment: str,
) -> tuple[str | None, str, str, str]:
    result = run_pipeline(
        video_path=video_path,
        profile_input={
            "goal": goal,
            "experience_level": experience_level,
            "intended_exercise": intended_exercise,
            "intended_variation": intended_variation or None,
            "known_limitations": limitations,
            "equipment": equipment,
        },
    )

    report = result["final_report"]
    video_quality = _quality_markdown(report["video_manifest"])
    mock_status = _mock_status_markdown(report)
    movement_metrics = _metrics_markdown(report)
    summary = report["coach_summary"]
    exercise = report["exercise"]
    variation = report["variation"]
    exercise_line = (
        f'{exercise["exercise"]} (mock confidence placeholder: {exercise["confidence"]:.0%})'
    )
    variation_line = (
        f'{variation["detected_variation"]} '
        f'(confidence: {variation["variation_confidence"]:.0%})'
    )
    finding = summary["main_findings"][0] if summary["main_findings"] else "No mock finding emitted"
    if not report["video_manifest"]["analysis_allowed"]:
        markdown = f"""{video_quality}

{mock_status}

## Run

- **Run ID:** `{report["run_id"]}`
- **Saved report:** `{Path(result["run_dir"]) / "final_report.json"}`
"""
        artifact_path = Path(result["run_dir"]) / "final_report.json"
        return (
            result["annotated_video_path"],
            markdown,
            _pretty_json(report),
            str(artifact_path),
        )

    markdown = f"""## Scan Summary

| Signal | Result |
| --- | --- |
| Exercise router | {report["exercise"]["exercise"]} ({report["exercise"]["confidence"]:.0%}) |
| Variation | {report["variation"]["detected_variation"]} ({report["variation"]["variation_confidence"]:.0%}) |
| Reps | {len(report["reps"]["reps"])} |
| Analysis mode | {report["artifacts"].get("analysis_mode", "unknown")} |
| Pose source | {report["artifacts"].get("pose_source", "unknown")} |
| Primary finding | {summary["main_findings"][0] if summary["main_findings"] else "No finding emitted"} |
| Run ID | `{report["run_id"]}` |

{movement_metrics}

{mock_status}

## Coach Notes

{summary["summary"]}

### What Went Well
{chr(10).join(f"- {item}" for item in summary["what_went_well"])}

### Top Fixes
{chr(10).join(f"- {item}" for item in summary["top_fixes"])}

### Next Session Plan
{chr(10).join(f"- {item}" for item in summary["next_session_plan"])}

{video_quality}
"""

    artifact_path = Path(result["run_dir"]) / "final_report.json"
    return (
        result["annotated_video_path"],
        markdown,
        _pretty_json(report),
        str(artifact_path),
    )


server = gr.Server(
    title="Pozify",
    summary="Video-based workout form review API",
    description=APP_DESCRIPTION,
)
server.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@server.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "pozify"}


@server.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@server.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "description": APP_DESCRIPTION,
        "goals": ["strength", "hypertrophy", "endurance", "mobility", "beginner_practice"],
        "experience_levels": ["beginner", "intermediate"],
        "exercises": ["auto", *USER_SELECTABLE_EXERCISES],
        "limitations": ["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
        "equipment": ["bodyweight", "dumbbell", "barbell", "unknown"],
    }


@server.get("/api/artifacts/{run_id}/{filename}", include_in_schema=False)
def artifact(run_id: str, filename: str) -> FileResponse:
    artifact_path = (RUNS_ROOT / run_id / filename).resolve()
    runs_root = RUNS_ROOT.resolve()
    if runs_root not in artifact_path.parents or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(artifact_path)


def _artifact_url(run_id: str, path: str | None) -> str | None:
    if not path:
        return None

    artifact_path = Path(path).resolve()
    run_root = (RUNS_ROOT / run_id).resolve()
    if run_root not in artifact_path.parents or not artifact_path.is_file():
        return None
    return f"/api/artifacts/{run_id}/{artifact_path.name}"


@server.post("/api/analyze")
async def analyze_api(
    video: UploadFile | None = File(default=None),
    goal: str = Form(default="beginner_practice"),
    experience_level: str = Form(default="beginner"),
    intended_exercise: str = Form(default="auto"),
    intended_variation: str = Form(default=""),
    limitations: str = Form(default="[]"),
    equipment: str = Form(default="bodyweight"),
) -> dict[str, Any]:
    try:
        parsed_limitations = json.loads(limitations)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Limitations must be valid JSON.") from exc

    if not isinstance(parsed_limitations, list) or not all(
        isinstance(item, str) for item in parsed_limitations
    ):
        raise HTTPException(status_code=400, detail="Limitations must be a JSON list of strings.")

    video_path: str | None = None
    if video is not None and video.filename:
        suffix = Path(video.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            shutil.copyfileobj(video.file, temp_video)
            video_path = temp_video.name

    try:
        result = run_pipeline(
            video_path=video_path,
            profile_input={
                "goal": goal,
                "experience_level": experience_level,
                "intended_exercise": intended_exercise,
                "intended_variation": intended_variation or None,
                "known_limitations": parsed_limitations,
                "equipment": equipment,
            },
        )
    finally:
        if video_path is not None:
            Path(video_path).unlink(missing_ok=True)

    return {
        "run_id": result["run_id"],
        "run_dir": result["run_dir"],
        "annotated_video_url": _artifact_url(result["run_id"], result["annotated_video_path"]),
        "final_report_url": f"/api/artifacts/{result['run_id']}/final_report.json",
        "report": result["final_report"],
    }


if __name__ == "__main__":
    server.launch(_frontend=False)
