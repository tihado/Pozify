from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from pozify.artifacts import write_json
from pozify.contracts import UserProfile, to_dict
from pozify.steps import (
    annotated_renderer,
    coach_summary,
    exercise_classifier,
    issue_marker,
    pose_cleaning,
    pose_landmarker,
    rep_analysis,
    rep_counter,
    variation_detector,
    verifier,
    video_qc,
)


RUNS_DIR = Path("runs")


def _env_mock_mode() -> bool:
    value = os.getenv("POZIFY_MOCK_MODE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def run_pipeline(
    video_path: str | None,
    profile_input: dict[str, Any],
    *,
    mock: bool | None = None,
) -> dict[str, Any]:
    mock_mode = _env_mock_mode() if mock is None else mock

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id
    artifact_index: list[dict[str, str]] = []

    def write_artifact(filename: str, payload: Any) -> None:
        path = write_json(run_dir, filename, payload)
        artifact_index.append(
            {
                "name": filename,
                "path": str(path),
                "contract": filename,
            }
        )

    profile = UserProfile(
        goal=profile_input["goal"],
        experience_level=profile_input["experience_level"],
        intended_exercise=profile_input.get("intended_exercise", "auto"),
        intended_variation=profile_input.get("intended_variation"),
        known_limitations=profile_input.get("known_limitations", []),
        equipment=profile_input.get("equipment", "unknown"),
    )
    write_artifact("user_profile.json", profile)

    manifest = video_qc.run(video_path)
    write_artifact("video_manifest.json", manifest)

    pose_sequence = pose_landmarker.run(manifest, mock=mock_mode)
    cleaned_pose_sequence = pose_cleaning.run(pose_sequence)
    write_artifact("pose_sequence.json", cleaned_pose_sequence)

    classification = exercise_classifier.run(cleaned_pose_sequence, profile, mock=mock_mode)
    write_artifact("exercise_classification.json", classification)

    reps, rep_debug = rep_counter.run(classification, cleaned_pose_sequence)
    write_artifact("reps.json", reps)
    write_artifact("rep_debug.json", rep_debug)

    analysis = rep_analysis.run(classification, reps, cleaned_pose_sequence)
    write_artifact("rep_analysis.json", analysis)

    variation = variation_detector.run(classification, analysis, profile)
    write_artifact("variation.json", variation)

    issues = issue_marker.run(classification, reps, analysis, variation)
    write_artifact("issue_markers.json", issues)

    annotated_video_path = annotated_renderer.run(manifest, cleaned_pose_sequence, reps, issues, run_dir)

    pose_source = (
        cleaned_pose_sequence.frames[0].pose_quality.get("source")
        if cleaned_pose_sequence.frames
        else "none"
    )
    analysis_mode = "mock" if mock_mode else "real"
    mock_steps = [
        "issue_marker",
        "coach_summary",
        "verifier",
    ]
    if mock_mode:
        mock_steps.insert(0, "exercise_classifier")

    summary = coach_summary.run(profile, classification, reps, analysis, variation, issues)
    write_artifact("coach_summary.json", summary)

    verification = verifier.run(summary, issues, variation)
    write_artifact("verification.json", verification)

    final_report = {
        "run_id": run_id,
        "profile": to_dict(profile),
        "video_manifest": to_dict(manifest),
        "exercise": to_dict(classification),
        "reps": to_dict(reps),
        "rep_analysis": to_dict(analysis),
        "variation": to_dict(variation),
        "issue_markers": to_dict(issues),
        "coach_summary": to_dict(summary),
        "verification": to_dict(verification),
        "artifacts": {
            "run_dir": str(run_dir),
            "annotated_video_path": annotated_video_path,
            "rep_debug_path": str(run_dir / "rep_debug.json"),
            "analysis_mode": analysis_mode,
            "pose_source": pose_source,
            "mock_steps": mock_steps,
        },
    }
    write_artifact("final_report.json", final_report)

    run_manifest = {
        "run_id": run_id,
        "mock_mode": mock_mode,
        "artifacts": artifact_index,
    }
    write_json(run_dir, "manifest.json", run_manifest)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "annotated_video_path": annotated_video_path,
        "manifest_path": str(run_dir / "manifest.json"),
        "final_report": final_report,
    }
