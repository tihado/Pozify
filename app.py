from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pozify.exercise_catalog import USER_SELECTABLE_EXERCISES
from pozify.pipeline import run_pipeline


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
    return f"""## Video Quality

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

- **Exercise router output:** {exercise_line}
- **Variation label:** {variation_line}
- **Reps:** {len(report["reps"]["reps"])}
- **Analysis mode:** {report["artifacts"].get("analysis_mode", "unknown")}
- **Pose source:** {report["artifacts"].get("pose_source", "unknown")}
- **Mock finding:** {finding}
- **Run ID:** `{report["run_id"]}`

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


with gr.Blocks(title="Pozify") as demo:
    gr.Markdown("# Pozify")
    gr.Markdown(
        "Upload a short workout video and run the full mocked review pipeline. "
        "All steps produce structured JSON artifacts that can be replaced with real models later."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Workout video", sources=["upload", "webcam"])
            goal = gr.Dropdown(
                label="Goal",
                choices=["strength", "hypertrophy", "endurance", "mobility", "beginner_practice"],
                value="beginner_practice",
            )
            experience_level = gr.Dropdown(
                label="Experience level",
                choices=["beginner", "intermediate"],
                value="beginner",
            )
            intended_exercise = gr.Dropdown(
                label="Intended exercise",
                choices=["auto", *USER_SELECTABLE_EXERCISES],
                value="auto",
            )
            intended_variation = gr.Textbox(
                label="Intended variation",
                placeholder="Optional, e.g. wide_grip_push_up",
            )
            limitations = gr.CheckboxGroup(
                label="Known limitations",
                choices=["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
                value=[],
            )
            equipment = gr.Dropdown(
                label="Equipment",
                choices=["bodyweight", "dumbbell", "barbell", "unknown"],
                value="bodyweight",
            )
            run_button = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            annotated_video = gr.Video(label="Annotated video")

    summary_md = gr.Markdown(label="Summary")

    with gr.Tab("Final Report JSON"):
        report_json = gr.Code(label="final_report.json", language="json", lines=28)

    with gr.Tab("Artifact Path"):
        artifact_path = gr.Textbox(label="Saved report")

    run_button.click(
        fn=analyze_video,
        inputs=[
            video,
            goal,
            experience_level,
            intended_exercise,
            intended_variation,
            limitations,
            equipment,
        ],
        outputs=[annotated_video, summary_md, report_json, artifact_path],
    )


if __name__ == "__main__":
    demo.launch()
