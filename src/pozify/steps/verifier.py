from __future__ import annotations

from pozify.contracts import CoachSummary, IssueMarkers, Variation, Verification


def run(summary: CoachSummary, issues: IssueMarkers, variation: Variation) -> Verification:
    issue_labels = {issue.issue for issue in issues.issues}
    mentioned_known_issue = (
        not issue_labels
        or any(label in " ".join(summary.main_findings + [summary.summary]) for label in issue_labels)
    )
    separated_variation = variation.detected_variation in summary.variation_explanation
    avoids_medical_claims = all(
        banned not in summary.summary.lower()
        for banned in ["diagnose", "injury prevention", "medical assessment"]
    )

    checks = {
        "mentions_only_known_issues_mock": mentioned_known_issue,
        "separates_variation_from_issue": separated_variation,
        "avoids_medical_claims": avoids_medical_claims,
        "includes_confidence_notes": bool(summary.confidence_notes),
    }

    return Verification(
        passed=all(checks.values()),
        checks=checks,
        notes=[] if all(checks.values()) else ["Mock verifier found a summary contract issue."],
    )

