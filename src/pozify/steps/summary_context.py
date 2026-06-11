from __future__ import annotations

from typing import Any

from pozify.contracts import (
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
)
from pozify.knowledge_cards import retrieve_cards


def _rep_trends(analysis: RepAnalysis) -> dict[str, float]:
    if len(analysis.items) < 2:
        return {
            "rom_delta": 0.0,
            "stability_delta": 0.0,
            "symmetry_delta": 0.0,
        }

    first = analysis.items[0]
    last = analysis.items[-1]
    return {
        "rom_delta": round(last.range_of_motion_score - first.range_of_motion_score, 2),
        "stability_delta": round(last.stability_score - first.stability_score, 2),
        "symmetry_delta": round(last.symmetry_score - first.symmetry_score, 2),
    }


def build_summary_context(
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    *,
    mock_steps: list[str] | None = None,
) -> dict[str, Any]:
    retrieved = retrieve_cards(
        exercise=classification.exercise,
        variation=variation.detected_variation,
        issues=[issue.issue for issue in issues.issues],
        goal=profile.goal,
    )
    return {
        "user_profile": {
            "goal": profile.goal,
            "experience_level": profile.experience_level,
            "intended_exercise": profile.intended_exercise,
            "intended_variation": profile.intended_variation,
            "known_limitations": list(profile.known_limitations),
            "equipment": profile.equipment,
        },
        "exercise": {
            "label": classification.exercise,
            "confidence": classification.confidence,
            "fallback_required": classification.fallback_required,
        },
        "rep_summary": {
            "rep_count": len(reps.reps),
            "partial_reps": list(reps.partial_reps),
            "aggregate_metrics": dict(analysis.aggregate_metrics),
            "trends": _rep_trends(analysis),
        },
        "variation": {
            "label": variation.detected_variation,
            "confidence": variation.variation_confidence,
            "not_issues": list(variation.not_issues),
        },
        "issues": [
            {
                "issue": issue.issue,
                "rep_id": issue.rep_id,
                "severity": issue.severity,
                "start_sec": issue.start_sec,
                "end_sec": issue.end_sec,
                "affected_joints": list(issue.affected_joints),
                "evidence": dict(issue.evidence),
            }
            for issue in issues.issues
        ],
        "knowledge_cards": [
            {
                "id": card.id,
                "type": card.type,
                "label": card.label,
                "summary": card.summary,
                "good_signals": list(card.good_signals),
                "common_misreads": list(card.common_misreads),
                "coaching_cues": list(card.coaching_cues),
                "safety_notes": list(card.safety_notes),
                "contraindicated_claims": list(card.contraindicated_claims),
            }
            for card in retrieved.cards
        ],
        "retrieval_trace": {
            "requested_labels": list(retrieved.trace.requested_labels),
            "matched_card_ids": list(retrieved.trace.matched_card_ids),
            "missing_labels": list(retrieved.trace.missing_labels),
        },
        "constraints": {
            "no_diagnosis": True,
            "no_injury_prevention_claim": True,
            "must_not_invent_issues": True,
            "must_separate_variation_from_issue": True,
        },
        "mock_steps": list(mock_steps or []),
    }
