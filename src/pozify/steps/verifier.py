from __future__ import annotations

import re

from pozify.contracts import CoachSummary, ExerciseClassification, IssueMarkers, Variation, Verification
from pozify.knowledge_cards import load_knowledge_cards


def _all_summary_text(summary: CoachSummary) -> str:
    return " ".join(
        [
            summary.summary,
            *summary.what_went_well,
            *summary.main_findings,
            summary.variation_explanation,
            *summary.top_fixes,
            *summary.next_session_plan,
            *summary.confidence_notes,
        ]
    ).lower()


def _issue_like_labels(text: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"\b[a-z]+(?:_[a-z]+)+\b", text.lower())}


def _safety_card_claims() -> dict[str, set[str]]:
    claims: dict[str, set[str]] = {}
    for card in load_knowledge_cards():
        if card.type != "safety":
            continue
        claims[card.label] = {claim.lower() for claim in card.contraindicated_claims}
    return claims


def run(
    summary: CoachSummary,
    issues: IssueMarkers,
    variation: Variation,
    classification: ExerciseClassification,
    *,
    mock_steps: list[str] | None = None,
) -> Verification:
    text = _all_summary_text(summary)
    issue_labels = {issue.issue for issue in issues.issues}
    safety_claims = _safety_card_claims()
    allowed_labels = {
        classification.exercise,
        variation.exercise,
        variation.detected_variation,
        *variation.not_issues,
        *issue_labels,
    }
    extraneous_issues = sorted(_issue_like_labels(text) - allowed_labels)

    mentions_only_known_issue = not extraneous_issues
    separated_variation = (
        variation.detected_variation in summary.variation_explanation
        and "not automatically" in summary.variation_explanation.lower()
    )
    avoids_diagnosis = all(
        banned not in text for banned in safety_claims.get("no_diagnosis", set())
    )
    avoids_injury_prevention = all(
        banned not in text for banned in safety_claims.get("no_injury_prevention_claim", set())
    )
    requires_confidence_notes = (
        classification.confidence < 0.95
        or bool(mock_steps)
        or not issues.issues
    )
    includes_confidence_notes = bool(summary.confidence_notes) if requires_confidence_notes else True
    avoids_overconfidence = all(
        banned not in text for banned in safety_claims.get("confidence_language", set())
    )

    checks = {
        "mentions_only_known_issues": mentions_only_known_issue,
        "separates_variation_from_issue": separated_variation,
        "avoids_diagnosis": avoids_diagnosis,
        "avoids_injury_prevention_claims": avoids_injury_prevention,
        "avoids_overconfident_language": avoids_overconfidence,
        "includes_confidence_notes_when_required": includes_confidence_notes,
    }
    notes: list[str] = []
    if extraneous_issues:
        notes.append(
            "Summary mentioned issue-like labels absent from issue_markers.json: "
            + ", ".join(extraneous_issues)
        )
    if not separated_variation:
        notes.append("Variation explanation did not clearly separate variation from true issues.")
    if not avoids_diagnosis:
        notes.append("Summary used diagnosis-style language that is not allowed.")
    if not avoids_injury_prevention:
        notes.append("Summary made an injury-prevention claim that is not allowed.")
    if not avoids_overconfidence:
        notes.append("Summary used overconfident language where uncertainty should be preserved.")
    if not includes_confidence_notes:
        notes.append("Confidence notes were required but missing.")

    return Verification(
        passed=all(checks.values()),
        checks=checks,
        notes=notes,
    )
