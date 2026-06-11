from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pozify.artifacts import write_json
from pozify.contracts import UserProfile, to_dict
from pozify.exercises import create_exercise_strategy
from pozify.steps import (
    annotated_renderer,
    coach_summary,
    exercise_classifier,
    pose_cleaning,
    pose_landmarker,
    verifier,
    video_qc,
)


RUNS_DIR = Path("runs")
ProgressCallback = Callable[[dict[str, Any]], None]


def _env_mock_mode(video_path: str | None) -> bool:
    configured = os.getenv("POZIFY_MOCK_MODE")
    if configured is None:
        return video_path is None

    value = configured.strip().lower()
    return value not in {"0", "false", "no", "off"}


def run_pipeline(
    video_path: str | None,
    profile_input: dict[str, Any],
    *,
    mock: bool | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    mock_mode = _env_mock_mode(video_path) if mock is None else mock

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
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

    def emit(step: str, status: str, text: str, **payload: Any) -> None:
        if progress is None:
            return
        progress(
            {
                "type": "progress",
                "step": step,
                "status": status,
                "text": text,
                "payload": payload,
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

    emit(
        "quality",
        "active",
        "First up, I am checking if the video is clear enough to coach from.",
    )
    manifest = video_qc.run(video_path)
    write_artifact("video_manifest.json", manifest)
    emit(
        "quality",
        "done",
        (
            "Quick note: the video has a few things to watch."
            if manifest.quality_warnings
            else "Nice, your video quality looks solid."
        ),
        warnings=manifest.quality_warnings,
        analysis_allowed=manifest.analysis_allowed,
    )

    emit(
        "pose",
        "active",
        "Now I am mapping your posture and tracking the key body landmarks.",
    )
    pose_sequence = pose_landmarker.run(manifest, mock=mock_mode)
    cleaned_pose_sequence = pose_cleaning.run(pose_sequence)
    write_artifact("pose_sequence.json", cleaned_pose_sequence)
    pose_source = (
        cleaned_pose_sequence.frames[0].pose_quality.get("source")
        if cleaned_pose_sequence.frames
        else "none"
    )
    emit(
        "pose",
        "done",
        "Posture tracking is done. I found the key landmarks I need.",
        frame_count=len(cleaned_pose_sequence.frames),
        pose_source=pose_source,
        pose_valid_ratio=cleaned_pose_sequence.pose_valid_ratio,
    )

    emit("exercise", "active", "Let me figure out which exercise you are doing.")
    classification = exercise_classifier.run(
        cleaned_pose_sequence, profile, mock=mock_mode
    )
    write_artifact("exercise_classification.json", classification)
    emit(
        "exercise",
        "done",
        f"Looks like you are doing {classification.exercise.replace('_', ' ')}.",
        exercise=classification.exercise,
        confidence=classification.confidence,
    )

    exercise = create_exercise_strategy(
        classification.exercise,
        video_manifest=manifest,
        pose_sequence=cleaned_pose_sequence,
        profile=profile,
    )

    emit("reps", "active", "Counting your reps now. One clean rep at a time.")
    reps, rep_debug = exercise.count()
    write_artifact("reps.json", reps)
    write_artifact("rep_debug.json", rep_debug)
    emit(
        "reps",
        "done",
        (
            f"I counted {len(reps.reps)} {classification.exercise.replace('_', ' ')} "
            "reps in this set."
        ),
        rep_count=len(reps.reps),
        exercise=classification.exercise,
    )

    emit(
        "issues",
        "active",
        "Almost there. I am checking the moments that may need a small fix.",
    )
    analysis = exercise.analyze_reps(reps)
    write_artifact("rep_analysis.json", analysis)

    variation = exercise.resolve_variation(analysis)
    write_artifact("variation.json", variation)

    issues = exercise.mark_issues(reps, analysis, variation)
    write_artifact("issue_markers.json", issues)
    emit(
        "issues",
        "done",
        (
            f"I found {len(issues.issues)} coaching point"
            f"{'' if len(issues.issues) == 1 else 's'} worth reviewing."
            if issues.issues
            else "Good news, I did not spot any clear form issues in this set."
        ),
        issue_count=len(issues.issues),
    )

    emit("render", "active", "I am preparing your annotated video and issue clips.")
    render_artifacts = annotated_renderer.run(
        manifest,
        cleaned_pose_sequence,
        reps,
        issues,
        run_dir,
    )
    emit(
        "render",
        "done",
        (
            "Your annotated video is ready."
            if render_artifacts.status == "ok"
            else "I could not render an annotated video, but the report is ready."
        ),
        annotated_video_path=render_artifacts.annotated_video_path,
        issue_clip_count=len(render_artifacts.issue_clip_paths),
        annotated_video_status=render_artifacts.status,
        annotated_video_reason=render_artifacts.reason,
    )

    emit(
        "coach",
        "active",
        "I am turning the scan into coaching notes you can use right away.",
    )
    analysis_mode = "mock" if mock_mode else "real"
    mock_steps: list[str] = []
    if mock_mode:
        mock_steps.insert(0, "exercise_classifier")

    draft_summary = coach_summary.run(
        profile, classification, reps, analysis, variation, issues, mock_steps=mock_steps
    )

    verification = verifier.run(
        draft_summary,
        issues,
        variation,
        classification,
        mock_steps=mock_steps,
    )
    if verification.passed:
        summary = draft_summary
    else:
        summary = coach_summary.build_fallback(
            profile,
            classification,
            reps,
            analysis,
            variation,
            issues,
            verification_notes=[*verification.notes, "Conservative fallback summary returned."],
            mock_steps=mock_steps,
        )
        verification = type(verification)(
            passed=False,
            checks=verification.checks,
            notes=[*verification.notes, "Conservative fallback summary returned."],
        )

    write_artifact("coach_summary.json", summary)
    write_artifact("verification.json", verification)
    emit(
        "coach",
        "done",
        "Coach notes are ready.",
        verification_passed=verification.passed,
    )

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
            "annotated_video_path": render_artifacts.annotated_video_path,
            "annotated_video_status": render_artifacts.status,
            "annotated_video_reason": render_artifacts.reason,
            "issue_thumbnail_paths": render_artifacts.issue_thumbnail_paths,
            "issue_clip_paths": render_artifacts.issue_clip_paths,
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
        "annotated_video_path": render_artifacts.annotated_video_path,
        "issue_thumbnail_paths": render_artifacts.issue_thumbnail_paths,
        "issue_clip_paths": render_artifacts.issue_clip_paths,
        "manifest_path": str(run_dir / "manifest.json"),
        "final_report": final_report,
    }
