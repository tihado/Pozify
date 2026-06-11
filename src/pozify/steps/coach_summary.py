from __future__ import annotations

from dataclasses import dataclass

from pozify.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    SummaryGeneration,
    UserProfile,
    Variation,
)
from pozify.steps.summary_context import build_summary_context
from pozify.steps.summary_provider import create_summary_provider


@dataclass(frozen=True)
class SummaryDraft:
    summary: CoachSummary | None
    generation: SummaryGeneration


def generate(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    mock_steps: list[str] | None = None,
) -> SummaryDraft:
    context = build_summary_context(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        mock_steps=mock_steps,
    )
    provider_result = create_summary_provider().generate(context)
    generation = SummaryGeneration(
        provider=provider_result.provider,
        backend=provider_result.backend,
        model=provider_result.model,
        prompt_contract_version=provider_result.prompt_contract_version,
        parse_ok=provider_result.parse_ok,
        parse_error=provider_result.parse_error,
        verifier_passed=None,
        fallback_used=False,
        raw_output_present=provider_result.raw_output is not None,
    )
    if provider_result.payload is None:
        return SummaryDraft(summary=None, generation=generation)
    payload = provider_result.payload
    summary = CoachSummary(
        summary=payload["summary"],
        what_went_well=list(payload["what_went_well"]),
        main_findings=list(payload["main_findings"]),
        variation_explanation=payload["variation_explanation"],
        top_fixes=list(payload["top_fixes"]),
        next_session_plan=list(payload["next_session_plan"]),
        confidence_notes=list(payload["confidence_notes"]),
    )
    return SummaryDraft(summary=summary, generation=generation)


def run(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    mock_steps: list[str] | None = None,
) -> CoachSummary:
    draft = generate(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        mock_steps=mock_steps,
    )
    if draft.summary is None:
        raise RuntimeError(draft.generation.parse_error or "Summary provider failed to generate a payload.")
    return draft.summary


def build_fallback(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    verification_notes: list[str],
    mock_steps: list[str] | None = None,
) -> CoachSummary:
    issue_labels = sorted({issue.issue for issue in issues.issues})
    issue_text = (
        "Observed issue labels: " + ", ".join(f"`{label}`" for label in issue_labels) + "."
        if issue_labels
        else "No issue labels were emitted from the current issue marker step."
    )
    return CoachSummary(
        summary=(
            f"The pipeline observed {len(reps.reps)} {classification.exercise} rep(s) and the variation label "
            f"`{variation.detected_variation}`. {issue_text}"
        ),
        what_went_well=[
            f"Average ROM score from the structured artifacts is {analysis.aggregate_metrics['avg_rom_score']}.",
            f"Average symmetry score from the structured artifacts is {analysis.aggregate_metrics['avg_symmetry_score']}.",
        ],
        main_findings=[
            issue_text,
            "A conservative fallback summary was used because the generated draft failed verification.",
        ],
        variation_explanation=(
            f"`{variation.detected_variation}` is treated as context and not automatically as an error."
        ),
        top_fixes=[
            "Keep the camera angle and setup consistent for the next review.",
            "Focus on one visible adjustment at a time.",
            "Use the issue timeline and aggregate metrics as the main comparison points.",
        ],
        next_session_plan=[
            "Repeat the same recording setup on the next set.",
            "Compare whether the same issue labels appear in the same part of the set.",
            "Use a slower tempo if you want cleaner evidence for the next comparison.",
        ],
        confidence_notes=[
            *verification_notes,
            f"Classifier confidence for this run is {classification.confidence:.0%}.",
            (
                "Some steps still use placeholders: " + ", ".join(mock_steps)
                if mock_steps
                else "Fallback summary stayed within structured evidence only."
            ),
        ],
    )
