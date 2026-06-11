from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from json import JSONDecodeError
import os
from typing import Any, Protocol
from pydantic import BaseModel, ConfigDict, ValidationError


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


class CoachSummaryPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    what_went_well: list[str]
    main_findings: list[str]
    variation_explanation: str
    top_fixes: list[str]
    next_session_plan: list[str]
    confidence_notes: list[str]


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def build_prompt_contract(context: dict[str, Any], *, compact: bool = False) -> str:
    output_schema = {
        "summary": "string",
        "what_went_well": ["string"],
        "main_findings": ["string"],
        "variation_explanation": "string",
        "top_fixes": ["string"],
        "next_session_plan": ["string"],
        "confidence_notes": ["string"],
    }
    rendered_schema = json.dumps(output_schema, ensure_ascii=False)
    rendered_context = json.dumps(
        context,
        ensure_ascii=False,
        separators=(",", ":"),
    ) if compact else json.dumps(context, ensure_ascii=False, indent=2)
    return (
        "You are generating a grounded coaching summary from structured evidence only.\n"
        "Allowed evidence: user_profile, exercise classification, variation, rep_summary, issues, and knowledge cards.\n"
        "Forbidden: diagnosing injuries, claiming injury prevention, inventing issues, inventing metrics, or reasoning directly from raw video.\n"
        "Variation labels are context, not automatic errors.\n"
        "If confidence is limited or steps are mocked, confidence_notes must say so.\n"
        "Return valid JSON only. Do not wrap the JSON in markdown fences. Do not add prose outside the JSON object.\n"
        "Return exactly one JSON object matching the required keys and value types.\n"
        f"Required output keys: {', '.join(REQUIRED_SUMMARY_KEYS)}.\n"
        f"Output schema: {rendered_schema}\n"
        f"Context:\n{rendered_context}"
    )


def _cards_by_type(context: dict[str, Any], card_type: str) -> list[dict[str, Any]]:
    return [card for card in context["knowledge_cards"] if card["type"] == card_type]


def _issue_summary_and_card(
    issues: list[dict[str, Any]],
    issue_cards: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if not issues:
        return "No issue interval labels were emitted from the current issue marker step.", None

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
    return issue_summary, top_issue_card


def _build_top_fixes(
    top_issue_card: dict[str, Any] | None,
    exercise_cards: list[dict[str, Any]],
    goal_cards: list[dict[str, Any]],
) -> list[str]:
    top_fixes: list[str] = []
    if top_issue_card is not None:
        top_fixes.extend(top_issue_card["coaching_cues"][:2])
    if not top_fixes and exercise_cards:
        top_fixes.extend(exercise_cards[0]["coaching_cues"][:2])
    if goal_cards:
        top_fixes.extend(goal_cards[0]["coaching_cues"][:1])
    return top_fixes[:3] or [
        "Keep the camera angle consistent so the next review is easier to compare.",
        "Use a slightly slower tempo on the next set to make each rep easier to inspect.",
    ]


def _build_confidence_notes(context: dict[str, Any]) -> list[str]:
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
    return confidence_notes


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
    try:
        validated = CoachSummaryPayloadModel.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Summary payload failed schema validation: {exc}") from exc
    return validated.model_dump()


def _structured_response_format() -> dict[str, Any]:
    schema = CoachSummaryPayloadModel.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "coach_summary",
            "schema": schema,
            "strict": True,
        },
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


def _compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        "issue": issue.get("issue"),
        "rep_id": issue.get("rep_id"),
        "severity": issue.get("severity"),
        "start_sec": issue.get("start_sec"),
        "end_sec": issue.get("end_sec"),
    }
    joints = issue.get("affected_joints")
    if isinstance(joints, list) and joints:
        compacted["affected_joints"] = joints[:3]
    evidence = issue.get("evidence")
    if isinstance(evidence, dict) and evidence:
        filtered = {}
        for key in ("confidence", "threshold", "mean_metric_value", "supporting_frames"):
            if key in evidence:
                filtered[key] = evidence[key]
        for key, value in evidence.items():
            if key in filtered:
                continue
            if isinstance(value, (int, float, str, bool)) and len(filtered) < 4:
                filtered[key] = value
        if filtered:
            compacted["evidence"] = filtered
    return compacted


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        "id": card.get("id"),
        "type": card.get("type"),
        "label": card.get("label"),
        "summary": card.get("summary"),
    }
    for key in ("coaching_cues", "common_misreads", "safety_notes", "contraindicated_claims"):
        value = card.get(key)
        if isinstance(value, list) and value:
            compacted[key] = value[:2]
    return compacted


def _trimmed_context(context: dict[str, Any], *, max_issues: int, max_cards: int) -> dict[str, Any]:
    trimmed = deepcopy(context)
    issues = trimmed.get("issues")
    if isinstance(issues, list):
        ranked = sorted(
            (issue for issue in issues if isinstance(issue, dict)),
            key=lambda issue: (
                -float(issue.get("severity", 0.0)),
                int(issue.get("rep_id", 0) or 0),
            ),
        )
        trimmed["issues"] = [_compact_issue(issue) for issue in ranked[:max_issues]]
        if len(ranked) > max_issues:
            counts: dict[str, int] = {}
            for issue in ranked:
                label = issue.get("issue")
                if isinstance(label, str):
                    counts[label] = counts.get(label, 0) + 1
            trimmed["issue_overview"] = {
                "total_intervals": len(ranked),
                "counts_by_label": counts,
                "truncated": True,
            }
    cards = trimmed.get("knowledge_cards")
    if isinstance(cards, list):
        compact_cards = [
            _compact_card(card) for card in cards if isinstance(card, dict)
        ]
        trimmed["knowledge_cards"] = compact_cards[:max_cards]
        if len(compact_cards) > max_cards:
            trimmed["retrieval_trace"] = {
                **trimmed.get("retrieval_trace", {}),
                "truncated_cards": len(compact_cards) - max_cards,
            }
    retrieval = trimmed.get("retrieval_trace")
    if isinstance(retrieval, dict):
        matched_ids = retrieval.get("matched_card_ids")
        if isinstance(matched_ids, list) and len(matched_ids) > max_cards:
            retrieval["matched_card_ids"] = matched_ids[:max_cards]
    return trimmed


def _prepare_context_for_backend(
    context: dict[str, Any],
    *,
    backend_name: str,
    context_window: int | None,
) -> tuple[dict[str, Any], bool]:
    if backend_name != "huggingface":
        return context, False

    target_chars = max(8000, (context_window or 8192) * 3)
    candidates = (
        (12, 8),
        (8, 6),
        (6, 5),
        (4, 4),
        (3, 3),
    )
    trimmed_any = False
    for max_issues, max_cards in candidates:
        candidate = _trimmed_context(context, max_issues=max_issues, max_cards=max_cards)
        prompt = build_prompt_contract(candidate, compact=True)
        trimmed_any = trimmed_any or candidate != context
        if len(prompt) <= target_chars:
            return candidate, trimmed_any

    fallback = _trimmed_context(context, max_issues=2, max_cards=2)
    return fallback, True


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
        issue_summary, top_issue_card = _issue_summary_and_card(issues, issue_cards)

        partial_rep_note = (
            f" The pipeline also flagged {len(partial_reps)} partial rep(s)."
            if partial_reps
            else ""
        )
        top_fixes = _build_top_fixes(top_issue_card, exercise_cards, goal_cards)
        confidence_notes = _build_confidence_notes(context)

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


class HuggingFaceCloudSummaryProvider:
    provider_name = "slm_cloud"
    backend_name = "huggingface"

    def __init__(self) -> None:
        self.model_name = os.getenv(
            "POZIFY_SUMMARY_CLOUD_MODEL",
            os.getenv("POZIFY_SUMMARY_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        )
        self.api_token = (
            os.getenv("POZIFY_SUMMARY_API_KEY")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.base_url = os.getenv("POZIFY_SUMMARY_BASE_URL")
        self.max_tokens = _env_int("POZIFY_SUMMARY_MAX_TOKENS", 512)
        self.temperature = _env_float("POZIFY_SUMMARY_TEMPERATURE", 0.2)

    def _client(self) -> Any:
        if not self.api_token:
            raise RuntimeError(
                "The Hugging Face cloud summary provider requires HF_TOKEN or "
                "POZIFY_SUMMARY_API_KEY."
            )
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face cloud summary provider requires huggingface_hub."
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": self.api_token}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        return InferenceClient(**client_kwargs)

    def _prompt(self, context: dict[str, Any]) -> str:
        prepared_context, _trimmed = _prepare_context_for_backend(
            context,
            backend_name=self.backend_name,
            context_window=16384,
        )
        return build_prompt_contract(prepared_context, compact=True)

    def _chat_completion(
        self,
        client: Any,
        prompt: str,
        *,
        structured: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce grounded coaching summaries from structured evidence. "
                        "Return exactly one JSON object and no extra prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if structured:
            kwargs["response_format"] = _structured_response_format()
        return client.chat.completions.create(**kwargs)

    def generate(self, context: dict[str, Any]) -> SummaryProviderResult:
        try:
            client = self._client()
            prompt = self._prompt(context)
            try:
                response = self._chat_completion(client, prompt, structured=True)
            except Exception as structured_exc:
                structured_error = str(structured_exc).lower()
                if not any(
                    fragment in structured_error
                    for fragment in (
                        "response_format",
                        "json_schema",
                        "structured output",
                        "json mode",
                        "not supported",
                        "unsupported",
                        "invalid_request_error",
                    )
                ):
                    raise
                response = self._chat_completion(client, prompt, structured=False)
        except Exception as exc:
            return SummaryProviderResult(
                payload=None,
                provider=self.provider_name,
                backend=self.backend_name,
                model=self.model_name,
                prompt_contract_version=PROMPT_CONTRACT_VERSION,
                parse_ok=False,
                parse_error=str(exc),
            )

        text = None
        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            return SummaryProviderResult(
                payload=None,
                provider=self.provider_name,
                backend=self.backend_name,
                model=self.model_name,
                prompt_contract_version=PROMPT_CONTRACT_VERSION,
                parse_ok=False,
                parse_error="Hugging Face cloud summary provider returned empty text.",
            )

        return _parse_slm_output(
            text,
            provider=self.provider_name,
            backend=self.backend_name,
            model=self.model_name,
        )


def create_summary_provider(name: str | None = None) -> SummaryProvider:
    provider_name = (name or os.getenv("POZIFY_SUMMARY_PROVIDER", "template")).strip().lower()
    if provider_name == "template":
        return TemplateSummaryProvider()
    if provider_name == "mock":
        return MockSummaryProvider()
    if provider_name == "unsafe_mock":
        return UnsafeMockSummaryProvider()
    if provider_name == "slm_cloud":
        return HuggingFaceCloudSummaryProvider()
    raise ValueError(f"Unknown summary provider: {provider_name!r}")
