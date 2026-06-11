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
        "goals": [
            "strength",
            "hypertrophy",
            "endurance",
            "mobility",
            "beginner_practice",
        ],
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
        raise HTTPException(
            status_code=400, detail="Limitations must be valid JSON."
        ) from exc

    if not isinstance(parsed_limitations, list) or not all(
        isinstance(item, str) for item in parsed_limitations
    ):
        raise HTTPException(
            status_code=400, detail="Limitations must be a JSON list of strings."
        )

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
        "annotated_video_url": _artifact_url(
            result["run_id"], result["annotated_video_path"]
        ),
        "final_report_url": f"/api/artifacts/{result['run_id']}/final_report.json",
        "report": result["final_report"],
    }


if __name__ == "__main__":
    server.launch(_frontend=False)
