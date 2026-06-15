from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from pozify.contracts import (
    CoachSummary,
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
    validate_contract,
)
from pozify.knowledge_cards import KnowledgeCard, retrieve_cards
from pozify.slm.prompting import build_coach_summary_prompt
from pozify.slm.providers import CoachSummaryModel, get_coach_summary_model
from pozify.steps.coach_summary_fallback import build_fallback_summary


@dataclass(frozen=True)
class CoachSummaryResult:
    summary: CoachSummary
    provider: str
    model: str
    source: str


def _text_preview(text: str, *, limit: int = 240) -> str:
    preview = " ".join(text.strip().split())
    if len(preview) > limit:
        return f"{preview[:limit]}..."
    return preview or "<empty>"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "Coach summary model output was not valid JSON. "
                f"Parser error: {exc}. Output preview: {_text_preview(text)}"
            ) from exc
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested_exc:
            raise ValueError(
                "Coach summary model output contained a JSON-like object that could not be "
                f"parsed. Parser error: {nested_exc}. Output preview: {_text_preview(text)}"
            ) from nested_exc

    if not isinstance(payload, dict):
        raise ValueError("Coach summary model output must be a JSON object")
    return payload


def _summary_from_payload(payload: dict[str, Any]) -> CoachSummary:
    summary = CoachSummary(
        summary=str(payload["summary"]),
        what_you_did=[str(item) for item in payload["what_you_did"]],
        what_looked_good=[str(item) for item in payload["what_looked_good"]],
        what_changed_across_reps=[str(item) for item in payload["what_changed_across_reps"]],
        valid_variation_vs_issue=[str(item) for item in payload["valid_variation_vs_issue"]],
        top_fixes=[str(item) for item in payload["top_fixes"]],
        next_session_plan=[str(item) for item in payload["next_session_plan"]],
        confidence_notes=[str(item) for item in payload["confidence_notes"]],
    )
    validate_contract("coach_summary.json", summary)
    return summary


def _fallback(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
    failure_reason: str | None = None,
) -> CoachSummary:
    return build_fallback_summary(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
        failure_reason=failure_reason,
    )


def run(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    cards: list[KnowledgeCard] | None = None,
    model: CoachSummaryModel | None = None,
) -> CoachSummary:
    return run_with_metadata(
        profile,
        classification,
        reps,
        analysis,
        variation,
        issues,
        cards=cards,
        model=model,
    ).summary


def run_with_metadata(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    cards: list[KnowledgeCard] | None = None,
    model: CoachSummaryModel | None = None,
) -> CoachSummaryResult:
    cards = cards or retrieve_cards(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    provider = get_coach_summary_model() if model is None else model
    if provider is None:
        return CoachSummaryResult(
            summary=_fallback(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=cards,
                failure_reason="remote provider unavailable",
            ),
            provider="none",
            model="none",
            source="fallback_initial",
        )

    prompt = build_coach_summary_prompt(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
    )

    try:
        generation = provider.generate_summary(prompt)
        payload = _extract_json_object(generation.text)
        return CoachSummaryResult(
            summary=_summary_from_payload(payload),
            provider=generation.provider,
            model=generation.model,
            source="model_or_local",
        )
    except Exception as exc:
        return CoachSummaryResult(
            summary=_fallback(
                profile=profile,
                classification=classification,
                reps=reps,
                analysis=analysis,
                variation=variation,
                issues=issues,
                cards=cards,
                failure_reason=str(exc),
            ),
            provider="fallback",
            model="fallback",
            source="fallback_initial",
        )
