from __future__ import annotations

import json
import os
from typing import Any, Protocol


class SummaryProvider(Protocol):
    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def build_prompt_contract(context: dict[str, Any]) -> str:
    return (
        "You are generating a grounded coaching summary from structured evidence only.\n"
        "Allowed evidence: user_profile, exercise classification, variation, rep_summary, issues, and knowledge cards.\n"
        "Forbidden: diagnosing injuries, claiming injury prevention, inventing issues, inventing metrics, or reasoning directly from raw video.\n"
        "Required output keys: summary, what_went_well, main_findings, variation_explanation, top_fixes, next_session_plan, confidence_notes.\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _cards_by_type(context: dict[str, Any], card_type: str) -> list[dict[str, Any]]:
    return [card for card in context["knowledge_cards"] if card["type"] == card_type]


class TemplateSummaryProvider:
    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        exercise = context["exercise"]["label"]
        rep_summary = context["rep_summary"]
        variation = context["variation"]
        issues = context["issues"]
        issue_cards = _cards_by_type(context, "issue")
        goal_cards = _cards_by_type(context, "goal")
        exercise_cards = _cards_by_type(context, "exercise")
        variation_cards = _cards_by_type(context, "variation")

        rep_count = rep_summary["rep_count"]
        partial_reps = rep_summary["partial_reps"]
        trends = rep_summary["trends"]
        aggregate_metrics = rep_summary["aggregate_metrics"]
        goal_summary = goal_cards[0]["summary"] if goal_cards else "Goal context is limited."
        exercise_summary = (
            exercise_cards[0]["summary"] if exercise_cards else f"The session was routed as {exercise}."
        )
        variation_summary = (
            variation_cards[0]["summary"]
            if variation_cards
            else f"The variation label for this run is `{variation['label']}`."
        )

        if issues:
            issue_counts: dict[str, int] = {}
            for issue in issues:
                issue_counts[issue["issue"]] = issue_counts.get(issue["issue"], 0) + 1
            main_issue, main_issue_count = sorted(
                issue_counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            issue_summary = (
                f"The main marked issue is `{main_issue}`, appearing in {main_issue_count} interval(s)."
            )
            top_issue_card = next((card for card in issue_cards if card["label"] == main_issue), None)
        else:
            main_issue = None
            issue_summary = "No issue interval labels were emitted from the current issue marker step."
            top_issue_card = None

        partial_rep_note = (
            f" The pipeline also flagged {len(partial_reps)} partial rep(s)."
            if partial_reps
            else ""
        )
        top_fixes = []
        if top_issue_card is not None:
            top_fixes.extend(top_issue_card["coaching_cues"][:2])
        if not top_fixes and exercise_cards:
            top_fixes.extend(exercise_cards[0]["coaching_cues"][:2])
        if goal_cards:
            top_fixes.extend(goal_cards[0]["coaching_cues"][:1])
        top_fixes = top_fixes[:3] or [
            "Keep the camera angle consistent so the next review is easier to compare.",
            "Use a slightly slower tempo on the next set to make each rep easier to inspect.",
        ]

        confidence_notes = []
        if context["mock_steps"]:
            confidence_notes.append(
                "Some downstream interpretation steps still use placeholders: "
                + ", ".join(context["mock_steps"])
                + "."
            )
        if context["exercise"]["confidence"] < 0.95:
            confidence_notes.append(
                f"Exercise routing confidence is {context['exercise']['confidence']:.0%}, so treat the coaching language conservatively."
            )
        if context["retrieval_trace"]["missing_labels"]:
            confidence_notes.append(
                "Some knowledge cards were not available for: "
                + ", ".join(context["retrieval_trace"]["missing_labels"])
                + "."
            )
        if not confidence_notes:
            confidence_notes.append("Summary grounded cleanly in the current structured artifacts.")

        return {
            "summary": (
                f"You performed {rep_count} {exercise.replace('_', ' ')} rep(s). "
                f"{exercise_summary} {issue_summary}{partial_rep_note} Goal context: {goal_summary}"
            ),
            "what_went_well": [
                f"Average range-of-motion score from the current artifact set is {aggregate_metrics.get('avg_rom_score', 0.0)}.",
                f"Average symmetry score from the current artifact set is {aggregate_metrics.get('avg_symmetry_score', 0.0)}.",
            ],
            "main_findings": [
                issue_summary,
                (
                    "Rep quality changed across the set with "
                    f"ROM delta {trends['rom_delta']}, stability delta {trends['stability_delta']}, "
                    f"and symmetry delta {trends['symmetry_delta']}."
                ),
            ],
            "variation_explanation": (
                f"{variation_summary} The variation label is `{variation['label']}`. "
                "It is used as context and is not automatically treated as an issue. "
                f"Known non-issue labels for this run: {', '.join(variation['not_issues']) or 'none'}."
            ),
            "top_fixes": top_fixes,
            "next_session_plan": [
                "Repeat the same camera angle and setup on the next set.",
                "Compare whether the same issue intervals appear at the same point in the set.",
                "Keep one main cue in focus instead of changing multiple things at once.",
            ],
            "confidence_notes": confidence_notes,
        }


class MockSummaryProvider:
    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        return TemplateSummaryProvider().generate(context)


class UnsafeMockSummaryProvider:
    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": "This definitely prevents injury and shows knee valgus that was not in the JSON.",
            "what_went_well": ["You moved with energy."],
            "main_findings": ["knee_valgus appeared throughout the set."],
            "variation_explanation": "The variation label means the movement was wrong.",
            "top_fixes": ["Fix the medical issue immediately."],
            "next_session_plan": ["Avoid injury by changing everything next session."],
            "confidence_notes": [],
        }


def create_summary_provider(name: str | None = None) -> SummaryProvider:
    provider_name = (name or os.getenv("POZIFY_SUMMARY_PROVIDER", "template")).strip().lower()
    if provider_name == "template":
        return TemplateSummaryProvider()
    if provider_name == "mock":
        return MockSummaryProvider()
    if provider_name == "unsafe_mock":
        return UnsafeMockSummaryProvider()
    raise ValueError(f"Unknown summary provider: {provider_name!r}")
