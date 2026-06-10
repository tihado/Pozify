from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pozify.pipeline import run_pipeline


def _pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


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
    summary = report["coach_summary"]
    markdown = f"""## Scan Summary

- **Exercise:** {report["exercise"]["exercise"]} ({report["exercise"]["confidence"]:.0%} confidence)
- **Variation:** {report["variation"]["detected_variation"]} ({report["variation"]["variation_confidence"]:.0%} confidence)
- **Reps:** {len(report["reps"]["reps"])}
- **Main finding:** {summary["main_findings"][0] if summary["main_findings"] else "No major issue detected"}
- **Run ID:** `{report["run_id"]}`

## Coach Notes

{summary["summary"]}

### What Went Well
{chr(10).join(f"- {item}" for item in summary["what_went_well"])}

### Top Fixes
{chr(10).join(f"- {item}" for item in summary["top_fixes"])}

### Next Session Plan
{chr(10).join(f"- {item}" for item in summary["next_session_plan"])}
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
                choices=["auto", "squat", "push_up", "shoulder_press"],
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

        with gr.Column(scale=2):
            annotated_video = gr.Video(label="Annotated video placeholder")
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
