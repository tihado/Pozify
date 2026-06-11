from __future__ import annotations

from dataclasses import dataclass
import json
from json import JSONDecodeError
import os
from typing import Any, Protocol

from pozify.steps.summary_slm_backend import create_summary_slm_backend


PROMPT_CONTRACT_VERSION = "v1"
REQUIRED_SUMMARY_KEYS = (
    "summary",
    "what_went_well",
    "main_findings",
    "variation_explanation",
    "top_fixes",
    "next_session_plan",
    "confidence_notes",
)


class SummaryProvider(Protocol):
    def generate(self, context: dict[str, Any]) -> "SummaryProviderResult":
        raise NotImplementedError


@dataclass(frozen=True)
class SummaryProviderResult:
    payload: dict[str, Any] | None
    provider: str
    backend: str | None
    model: str | None
    prompt_contract_version: str
    parse_ok: bool
    parse_error: str | None = None
    raw_output: str | None = None


def build_prompt_contract(context: dict[str, Any]) -> str:
    output_schema = {
        "summary": "string",
        "what_went_well": ["string"],
        "main_findings": ["string"],
        "variation_explanation": "string",
        "top_fixes": ["string"],
        "next_session_plan": ["string"],
        "confidence_notes": ["string"],
    }
    return (
        "You are generating a grounded coaching summary from structured evidence only.\n"
        "Allowed evidence: user_profile, exercise classification, variation, rep_summary, issues, and knowledge cards.\n"
        "Forbidden: diagnosing injuries, claiming injury prevention, inventing issues, inventing metrics, or reasoning directly from raw video.\n"
        "Variation labels are context, not automatic errors.\n"
        "If confidence is limited or steps are mocked, confidence_notes must say so.\n"
        "Return valid JSON only. Do not wrap the JSON in markdown fences. Do not add prose outside the JSON object.\n"
        f"Required output keys: {', '.join(REQUIRED_SUMMARY_KEYS)}.\n"
        f"Output schema: {json.dumps(output_schema, ensure_ascii=False)}\n"
        f"Context:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )


def _cards_by_type(context: dict[str, Any], card_type: str) -> list[dict[str, Any]]:
    return [card for card in context["knowledge_cards"] if card["type"] == card_type]


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Model output did not contain a valid JSON object.")


def _validate_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(REQUIRED_SUMMARY_KEYS) - payload.keys())
    if missing:
        raise ValueError(f"Summary payload missing required keys: {', '.join(missing)}")

    if not isinstance(payload["summary"], str):
        raise ValueError("summary must be a string")
    if not isinstance(payload["variation_explanation"], str):
        raise ValueError("variation_explanation must be a string")

    for key in (
        "what_went_well",
        "main_findings",
        "top_fixes",
        "next_session_plan",
        "confidence_notes",
    ):
        value = payload[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must be a list of strings")

    return {
        "summary": payload["summary"],
        "what_went_well": list(payload["what_went_well"]),
        "main_findings": list(payload["main_findings"]),
        "variation_explanation": payload["variation_explanation"],
        "top_fixes": list(payload["top_fixes"]),
        "next_session_plan": list(payload["next_session_plan"]),
        "confidence_notes": list(payload["confidence_notes"]),
    }


def _parse_slm_output(
    raw_output: str,
    *,
    provider: str,
    backend: str,
    model: str,
) -> SummaryProviderResult:
    try:
        payload = _validate_summary_payload(_extract_json_object(raw_output))
    except ValueError as exc:
        return SummaryProviderResult(
            payload=None,
            provider=provider,
            backend=backend,
            model=model,
            prompt_contract_version=PROMPT_CONTRACT_VERSION,
            parse_ok=False,
            parse_error=str(exc),
            raw_output=raw_output,
        )

    return SummaryProviderResult(
        payload=payload,
        provider=provider,
        backend=backend,
        model=model,
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        parse_ok=True,
        raw_output=raw_output,
    )


class TemplateSummaryProvider:
    provider_name = "template"

    def generate(self, context: dict[str, Any]) -> SummaryProviderResult:
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

        payload = {
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
        return SummaryProviderResult(
            payload=payload,
            provider=self.provider_name,
            backend=None,
            model=None,
            prompt_contract_version=PROMPT_CONTRACT_VERSION,
            parse_ok=True,
        )


class MockSummaryProvider:
    provider_name = "mock"

    def generate(self, context: dict[str, Any]) -> SummaryProviderResult:
        result = TemplateSummaryProvider().generate(context)
        return SummaryProviderResult(
            payload=result.payload,
            provider=self.provider_name,
            backend=result.backend,
            model=result.model,
            prompt_contract_version=result.prompt_contract_version,
            parse_ok=result.parse_ok,
            parse_error=result.parse_error,
            raw_output=result.raw_output,
        )


class UnsafeMockSummaryProvider:
    provider_name = "unsafe_mock"

    def generate(self, context: dict[str, Any]) -> SummaryProviderResult:
        payload = {
            "summary": "This definitely prevents injury and shows knee_valgus that was not in the JSON.",
            "what_went_well": ["You moved with energy."],
            "main_findings": ["knee_valgus appeared throughout the set."],
            "variation_explanation": "The variation label means the movement was wrong.",
            "top_fixes": ["Fix the medical issue immediately."],
            "next_session_plan": ["Avoid injury by changing everything next session."],
            "confidence_notes": [],
        }
        return SummaryProviderResult(
            payload=payload,
            provider=self.provider_name,
            backend=None,
            model=None,
            prompt_contract_version=PROMPT_CONTRACT_VERSION,
            parse_ok=True,
        )


class OpenSourceSlmProvider:
    provider_name = "slm_local"

    def generate(self, context: dict[str, Any]) -> SummaryProviderResult:
        backend = create_summary_slm_backend()
        prompt = build_prompt_contract(context)
        try:
            backend_result = backend.generate_text(prompt)
        except Exception as exc:
            return SummaryProviderResult(
                payload=None,
                provider=self.provider_name,
                backend=getattr(backend, "backend_name", "unknown"),
                model=getattr(backend, "model_name", None),
                prompt_contract_version=PROMPT_CONTRACT_VERSION,
                parse_ok=False,
                parse_error=str(exc),
            )
        return _parse_slm_output(
            backend_result.text,
            provider=self.provider_name,
            backend=backend_result.backend,
            model=backend_result.model,
        )


def create_summary_provider(name: str | None = None) -> SummaryProvider:
    provider_name = (name or os.getenv("POZIFY_SUMMARY_PROVIDER", "template")).strip().lower()
    if provider_name == "template":
        return TemplateSummaryProvider()
    if provider_name == "mock":
        return MockSummaryProvider()
    if provider_name == "unsafe_mock":
        return UnsafeMockSummaryProvider()
    if provider_name == "slm_local":
        return OpenSourceSlmProvider()
    raise ValueError(f"Unknown summary provider: {provider_name!r}")
