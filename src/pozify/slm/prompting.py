from __future__ import annotations

import json
from typing import Any

from pozify.contracts import (
    ExerciseClassification,
    IssueMarkers,
    RepAnalysis,
    Reps,
    UserProfile,
    Variation,
    to_dict,
)
from pozify.knowledge_cards import KnowledgeCard, prioritized_coaching_points


def build_summary_evidence(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    top_issue_intervals = []
    for issue in sorted(issues.issues, key=lambda item: item.severity, reverse=True):
        issue_counts[issue.issue] = issue_counts.get(issue.issue, 0) + 1
        if len(top_issue_intervals) < 3:
            top_issue_intervals.append(
                {
                    "issue": issue.issue,
                    "rep_id": issue.rep_id,
                    "severity": round(issue.severity, 3),
                    "start_sec": round(issue.start_sec, 3),
                    "end_sec": round(issue.end_sec, 3),
                    "evidence": issue.evidence,
                }
            )

    return {
        "user_profile": to_dict(profile),
        "exercise_classification": {
            "exercise": classification.exercise,
            "confidence": classification.confidence,
            "fallback_required": classification.fallback_required,
        },
        "variation": to_dict(variation),
        "rep_summary": {
            "rep_count": len(reps.reps),
            "aggregate_metrics": analysis.aggregate_metrics,
            "rep_metrics": [to_dict(item) for item in analysis.items],
        },
        "issue_summary": {
            "issue_counts": issue_counts,
            "issues": [to_dict(issue) for issue in issues.issues],
            "top_issue_intervals": top_issue_intervals,
        },
        "priority_cues": prioritized_coaching_points(cards),
        "knowledge_cards": [to_dict(card) for card in cards],
    }


def build_coach_summary_prompt(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    reps: Reps,
    analysis: RepAnalysis,
    variation: Variation,
    issues: IssueMarkers,
    cards: list[KnowledgeCard],
) -> str:
    evidence = build_summary_evidence(
        profile=profile,
        classification=classification,
        reps=reps,
        analysis=analysis,
        variation=variation,
        issues=issues,
        cards=cards,
    )
    expected_schema = {
        "summary": "string",
        "what_you_did": ["string"],
        "what_looked_good": ["string"],
        "what_changed_across_reps": ["string"],
        "valid_variation_vs_issue": ["string"],
        "top_fixes": ["string"],
        "next_session_plan": ["string"],
        "confidence_notes": ["string"],
    }

    instructions = {
        "response_format": {
            "type": "json_object",
            "schema": expected_schema,
        },
        "task": "Generate a grounded coach summary from structured exercise-analysis artifacts.",
        "rules": [
            "Return exactly one JSON object that matches response_format.schema.",
            "The first output character must be `{`.",
            "Use only the evidence JSON and retrieved knowledge cards.",
            "Do not infer new issues that are absent from issue_summary.issues.",
            "Do not diagnose injuries, pain, mobility deficits, or pathology.",
            "Do not claim injury prevention.",
            "Do not treat a valid detected variation or not-issue label as an error.",
            "Use priority_cues as the first source for phrasing top fixes and next-session guidance.",
            "When referencing an issue or variation, include its exact label in "
            "backticks at least once.",
            "If evidence is limited or confidence is low, say so in confidence_notes.",
            "Return JSON only. No markdown fences. No extra commentary.",
        ],
    }
    json_dump_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
    return "\n\n".join(
        [
            json.dumps(instructions, **json_dump_kwargs),
            json.dumps(evidence, **json_dump_kwargs),
        ]
    )
