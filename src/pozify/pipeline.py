from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pozify import sample_pose_cache
from pozify.artifacts import write_json
from pozify.contracts import UserProfile, to_dict
from pozify.env import env_truthy, load_local_env
from pozify.exercises import create_exercise_strategy
from pozify.knowledge_cards import retrieve_cards
from pozify.steps import (
    annotated_renderer,
    coach_summary,
    coach_summary_fallback,
    exercise_classifier,
    pose_cleaning,
    pose_landmarker,
    verifier,
    video_qc,
)


RUNS_DIR = Path("runs")
ProgressCallback = Callable[[dict[str, Any]], None]
BYPASS_VERIFIER_ENV = "POZIFY_COACH_SUMMARY_BYPASS_VERIFIER"


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
    bypass_verifier: bool | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    load_local_env()
    mock_mode = _env_mock_mode(video_path) if mock is None else mock
    bypass_verifier_enabled = (
        env_truthy(os.getenv(BYPASS_VERIFIER_ENV))
        if bypass_verifier is None
        else bypass_verifier
    )

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
    cached_pose_sequence = None if mock_mode else sample_pose_cache.load(manifest)
    if cached_pose_sequence is None:
        pose_sequence = pose_landmarker.run(manifest, mock=mock_mode)
        cleaned_pose_sequence = pose_cleaning.run(pose_sequence)
    else:
        cleaned_pose_sequence = cached_pose_sequence
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
        pose_cache_hit=cached_pose_sequence is not None,
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
            if render_artifacts.annotated_video_path
            else "I could not render an annotated video, but the report is ready."
        ),
        annotated_video_path=render_artifacts.annotated_video_path,
        issue_clip_count=len(render_artifacts.issue_clip_paths),
    )

    emit(
        "coach",
        "active",
        "I am turning the scan into coaching notes you can use right away.",
    )
    analysis_mode = "mock" if mock_mode else "real"
    mock_steps = [
        "coach_summary",
        "verifier",
    ]
    if mock_mode:
        mock_steps.insert(0, "exercise_classifier")

    summary_cards = retrieve_cards(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    coach_result = coach_summary.run_with_metadata(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        cards=summary_cards,
    )
    summary = coach_result.summary
    verification = verifier.run(
        summary,
        issues,
        variation,
        classification=classification,
        analysis=analysis,
        reps=reps,
    )
    coach_summary_source = coach_result.source
    coach_summary_provider = coach_result.provider
    coach_summary_model = coach_result.model
    coach_summary_verifier_bypassed = False
    if not verification.passed:
        if bypass_verifier_enabled and coach_result.source == "model_or_local":
            coach_summary_verifier_bypassed = True
        else:
            summary = coach_summary_fallback.build_fallback_summary(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=summary_cards,
                failure_reason="; ".join(verification.notes) or "verification_failed",
            )
            coach_summary_source = "fallback_after_verification"
            coach_summary_provider = coach_result.provider
            coach_summary_model = coach_result.model
            verification = verifier.run(
                summary,
                issues,
                variation,
                classification=classification,
                analysis=analysis,
                reps=reps,
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
            "issue_thumbnail_paths": render_artifacts.issue_thumbnail_paths,
            "issue_clip_paths": render_artifacts.issue_clip_paths,
            "rep_debug_path": str(run_dir / "rep_debug.json"),
            "analysis_mode": analysis_mode,
            "pose_source": pose_source,
            "mock_steps": mock_steps,
            "coach_summary_source": coach_summary_source,
            "coach_summary_provider": coach_summary_provider,
            "coach_summary_model": coach_summary_model,
            "coach_summary_verifier_bypassed": coach_summary_verifier_bypassed,
            "coach_summary_verifier_bypass_requested": bypass_verifier_enabled,
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
