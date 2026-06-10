from __future__ import annotations

from datetime import datetime, timezone
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


def run_pipeline(video_path: str | None, profile_input: dict[str, Any]) -> dict[str, Any]:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = RUNS_DIR / run_id

    profile = UserProfile(
        goal=profile_input["goal"],
        experience_level=profile_input["experience_level"],
        intended_exercise=profile_input.get("intended_exercise", "auto"),
        intended_variation=profile_input.get("intended_variation"),
        known_limitations=profile_input.get("known_limitations", []),
        equipment=profile_input.get("equipment", "unknown"),
    )
    write_json(run_dir, "user_profile.json", profile)

    manifest = video_qc.run(video_path)
    write_json(run_dir, "video_manifest.json", manifest)

    pose_sequence = pose_landmarker.run(manifest)
    cleaned_pose_sequence = pose_cleaning.run(pose_sequence)
    write_json(run_dir, "pose_sequence.json", cleaned_pose_sequence)

    classification = exercise_classifier.run(cleaned_pose_sequence, profile)
    write_json(run_dir, "exercise_classification.json", classification)

    reps = rep_counter.run(classification, cleaned_pose_sequence)
    write_json(run_dir, "reps.json", reps)

    analysis = rep_analysis.run(classification, reps, cleaned_pose_sequence)
    write_json(run_dir, "rep_analysis.json", analysis)

    variation = variation_detector.run(classification, analysis, profile)
    write_json(run_dir, "variation.json", variation)

    issues = issue_marker.run(classification, reps, analysis, variation)
    write_json(run_dir, "issue_markers.json", issues)

    annotated_video_path = annotated_renderer.run(manifest, issues, run_dir)

    summary = coach_summary.run(profile, classification, reps, analysis, variation, issues)
    write_json(run_dir, "coach_summary.json", summary)

    verification = verifier.run(summary, issues, variation)
    write_json(run_dir, "verification.json", verification)

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
        },
    }
    write_json(run_dir, "final_report.json", final_report)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "annotated_video_path": annotated_video_path,
        "final_report": final_report,
    }

