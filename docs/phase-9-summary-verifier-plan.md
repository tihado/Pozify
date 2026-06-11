# Phase 9 Plan: Grounded Coach Summary And Verifier

## Branch

- Working branch: `ht/summary-verifier`

## Objective

Implement grounded coach summary generation with deterministic knowledge-card retrieval and a verifier that can reject unsafe or ungrounded summaries.

The summary layer must explain structured evidence from pipeline artifacts. It must not:

- infer directly from raw video
- invent metrics or issues
- diagnose injuries
- claim injury prevention
- treat valid variation as an error

## Current Codebase Assessment

### What already exists

- Pipeline integration points for summary and verification are already in place in [src/pozify/pipeline.py](/Users/h/Documents/Pozify/src/pozify/pipeline.py:212).
- `CoachSummary` and `Verification` contracts already exist in [src/pozify/contracts.py](/Users/h/Documents/Pozify/src/pozify/contracts.py:134).
- UI already displays:
  - summary
  - what went well
  - fixes
  - next-session plan
  - confidence notes
- `final_report.json` already carries `coach_summary` and `verification`.

### What is still missing

- No knowledge-card data model exists yet.
- No deterministic retrieval layer exists.
- No model provider abstraction exists for summary generation.
- `coach_summary.py` is still placeholder text assembly and explicitly says parts are mocked.
- `verifier.py` only performs shallow checks:
  - issue label mention
  - variation mention
  - a very small banned phrase list
  - confidence note presence
- No fallback summary policy is implemented beyond trivial pass/fail.
- No tests exist for grounded summary behavior, retrieval, or verifier failure modes.

### Main architectural gap

The repo already has structured evidence artifacts, but there is no intermediate layer that translates those artifacts into:

1. deterministic domain knowledge retrieval
2. a strict prompt contract
3. a provider interface
4. post-generation verification and fallback

That missing middle layer is the core of this issue.

## Proposed Solution

Implement the summary system as four explicit layers:

1. `knowledge_cards`
   - static, versioned domain knowledge
2. `summary_context_builder`
   - deterministic retrieval + compact prompt payload creation
3. `summary_provider`
   - model interface and provider adapters
4. `summary_verifier`
   - rule-based grounding and safety validation with fallback

This keeps the system explainable and testable even before adding a real SLM provider.

## Recommended File Layout

- `src/pozify/knowledge_cards/__init__.py`
- `src/pozify/knowledge_cards/schema.py`
- `src/pozify/knowledge_cards/loader.py`
- `src/pozify/knowledge_cards/retrieval.py`
- `src/pozify/knowledge_cards/cards/exercises.json`
- `src/pozify/knowledge_cards/cards/variations.json`
- `src/pozify/knowledge_cards/cards/issues.json`
- `src/pozify/knowledge_cards/cards/goals.json`
- `src/pozify/knowledge_cards/cards/safety.json`
- `src/pozify/steps/summary_provider.py`
- `src/pozify/steps/summary_context.py`
- `src/pozify/steps/coach_summary.py`
- `src/pozify/steps/verifier.py`
- `tests/test_knowledge_cards.py`
- `tests/test_summary_context.py`
- `tests/test_coach_summary.py`
- `tests/test_verifier.py`

## Workstream Plan

### 1. Knowledge Card System

Goal:

- Create a structured knowledge base that can be retrieved by labels only.

Implementation:

- Define a strict card schema, for example:
  - `id`
  - `type`
  - `label`
  - `aliases`
  - `summary`
  - `good_signals`
  - `common_misreads`
  - `coaching_cues`
  - `safety_notes`
  - `contraindicated_claims`
- Separate card collections by domain:
  - exercises
  - variations
  - issues
  - goals
  - safety rules
- Store starter cards as repo-local JSON files.

Required starter cards:

- exercises:
  - `squat`
  - `push_up`
  - `shoulder_press`
- variations:
  - `wide_grip_push_up`
  - `knee_push_up`
- issues:
  - `shallow_depth`
  - `hip_sag`
  - `incomplete_lockout`
- goals:
  - at least one card per current `goal` enum
- safety:
  - no diagnosis
  - no injury-prevention promises
  - uncertainty language guidance

Deliverables:

- Card schema
- Static card files
- Loader with validation

### 2. Deterministic Retrieval

Goal:

- Retrieve knowledge strictly from labels already present in structured artifacts.

Implementation:

- Build a retrieval function that accepts:
  - `profile.goal`
  - `classification.exercise`
  - `variation.detected_variation`
  - `issue_markers[*].issue`
- Retrieval must not use fuzzy search for MVP.
- Priority order:
  1. exercise card
  2. variation card
  3. unique issue cards
  4. goal card
  5. safety cards
- Deduplicate cards by `id`.
- Return cards in deterministic order for reproducibility.

Deliverables:

- `retrieve_cards(...) -> list[KnowledgeCard]`
- Small retrieval trace for debugging, for example:
  - requested labels
  - matched card ids
  - missing labels

### 3. Summary Context Builder

Goal:

- Build a compact, provider-agnostic context payload for generation.

Implementation:

- Add a context builder that converts artifacts into a strict summary input object.
- Include:
  - user profile
  - exercise classification
  - rep count
  - aggregate metrics
  - trend cues across reps
  - variation
  - issue intervals
  - retrieved card excerpts
- Exclude:
  - raw video
  - unconstrained freeform text
- Explicitly mark uncertain fields and mocked steps when relevant.

Recommended context shape:

```json
{
  "user_profile": {},
  "exercise": {},
  "rep_summary": {},
  "variation": {},
  "issues": [],
  "knowledge_cards": [],
  "constraints": {
    "no_diagnosis": true,
    "no_injury_prevention_claim": true,
    "must_not_invent_issues": true
  }
}
```

Deliverables:

- `build_summary_context(...)`
- Stable serialized payload for prompt generation and tests

### 4. Provider Interface

Goal:

- Hide model calls behind a swappable interface.

Implementation:

- Add an interface like:
  - `SummaryProvider.generate(context) -> CoachSummaryDraft`
- Provide two implementations initially:
  - `MockSummaryProvider`
  - `TemplateSummaryProvider`
- Delay remote/API-backed provider until the contract is stable.
- Keep provider selection behind env/config:
  - `POZIFY_SUMMARY_PROVIDER=mock|template|openai|hf|...`

Why this approach:

- It lets the team finish grounding, verification, and UI flow before introducing networked model calls.
- It keeps tests deterministic.

Deliverables:

- provider interface
- default local provider
- configuration path in `coach_summary.py`

### 5. Prompt Contract

Goal:

- Create a generation contract that limits the model to structured evidence.

Implementation:

- Generate a prompt with sections:
  - task
  - allowed evidence
  - forbidden behaviors
  - required output schema
- Explicitly instruct the model:
  - do not infer from video directly
  - do not mention issue labels absent from JSON
  - do not treat variation as error unless issue evidence supports it
  - do not diagnose injuries
  - include confidence notes when evidence is weak or some steps are mocked
- Require output fields to map to `CoachSummary`:
  - `summary`
  - `what_went_well`
  - `main_findings`
  - `variation_explanation`
  - `top_fixes`
  - `next_session_plan`
  - `confidence_notes`

Deliverables:

- Prompt builder
- Structured output parsing
- Local prompt fixtures in tests

### 6. Verifier

Goal:

- Reject ungrounded or unsafe summaries before they reach the user.

Implementation:

- Expand `verifier.py` to run explicit checks:
  - no issue outside `issue_markers.json`
  - variation not overcorrected
  - no diagnosis terms
  - no injury-prevention claims
  - no claims stronger than evidence supports
  - confidence notes present when:
    - classifier low confidence
    - no issues found
    - some steps are mocked
    - pose quality is weak
- Return granular checks in `Verification.checks`.
- Return human-readable failure reasons in `Verification.notes`.

Recommended checks:

- `mentions_only_known_issues`
- `separates_variation_from_issue`
- `avoids_diagnosis`
- `avoids_injury_prevention_claims`
- `includes_confidence_notes_when_required`
- `stays_within_known_metrics`

Deliverables:

- stronger verifier
- machine-readable check names
- actionable failure notes

### 7. Conservative Fallback Summary

Goal:

- Show a safe, grounded fallback when generated output fails verification.

Implementation:

- Add a deterministic fallback summary generator that only uses:
  - exercise label
  - rep count
  - variation label
  - issue labels
  - aggregate metrics
  - explicit uncertainty notes
- Fallback should avoid:
  - speculative interpretation
  - strong corrective language
  - diagnosis
- Mark fallback clearly in `confidence_notes`.

Example fallback behavior:

- If verification fails, use:
  - brief summary of what the pipeline observed
  - list of issue labels only
  - conservative next step
  - explicit uncertainty note

Deliverables:

- deterministic fallback builder
- pipeline path:
  - generate draft
  - verify
  - fallback if failed

## Concrete Implementation Order

1. Create card schema and static JSON card files.
2. Add card loader and validation.
3. Add deterministic retrieval.
4. Add summary context builder.
5. Add provider interface with local template/mock provider.
6. Rewrite `coach_summary.py` to:
   - build context
   - retrieve cards
   - call provider
   - parse output
7. Expand `verifier.py` checks.
8. Add fallback summary path.
9. Surface verifier/fallback state in UI and final report if needed.
10. Add tests across retrieval, summary generation, verifier, and fallback.

## Test Plan

### Unit Tests

#### Knowledge cards

- Card files load successfully.
- Required fields exist.
- Duplicate card ids fail validation.
- Retrieval returns deterministic order.
- Missing labels are handled cleanly.

#### Summary context

- Context includes only structured inputs.
- Context excludes raw video/direct inference paths.
- Context includes all required cards.

#### Provider / summary generation

- Template/mock provider returns valid `CoachSummary`.
- Output parsing rejects malformed provider output.

#### Verifier

- Fails when summary mentions an unknown issue.
- Fails when variation is described as an error without issue support.
- Fails on diagnosis language.
- Fails on injury-prevention claims.
- Fails when confidence notes are required but missing.

#### Fallback

- Fallback is used when verification fails.
- Fallback output still satisfies `CoachSummary` contract.

### Integration Tests

- Full pipeline writes:
  - `coach_summary.json`
  - `verification.json`
- `verification.passed=true` for grounded template summary.
- `verification.passed=false` triggers fallback summary.
- Final report includes fallback summary when draft is rejected.

### Scenario Tests

- squat + shallow depth + beginner goal
- push-up + wide-grip variation + no true issue
- shoulder press + incomplete lockout
- unknown exercise or low-confidence case
- low-pose-quality run requiring explicit confidence note

## Acceptance Criteria Mapping

- Coach summary is generated from structured artifacts.
  - satisfied by context builder + provider contract
- Knowledge cards are retrieved deterministically.
  - satisfied by label-only retrieval
- Summary does not mention issues absent from JSON.
  - satisfied by verifier + tests
- Verifier can fail unsafe or ungrounded summaries.
  - satisfied by expanded checks
- Conservative fallback is shown when verification fails.
  - satisfied by fallback path in `coach_summary.py`
- UI displays summary, fixes, plan, and confidence notes.
  - already mostly in place; verify wording and fallback rendering

## Risks And Mitigations

### Risk: overbuilding model integration too early

Mitigation:

- Start with template/mock provider and prompt contract first.

### Risk: retrieval becomes fuzzy and non-deterministic

Mitigation:

- Restrict MVP retrieval to exact label mapping only.

### Risk: verifier too weak

Mitigation:

- Encode explicit banned patterns and evidence-based checks.

### Risk: verifier too strict and rejects everything

Mitigation:

- Add granular notes so failures are debuggable.
- Start with a template provider to establish a passing baseline.

## Recommended First PR Breakdown

### PR 1

- card schema
- starter cards
- loader
- retrieval tests

### PR 2

- summary context builder
- provider interface
- template/mock provider

### PR 3

- expanded verifier
- fallback summary
- pipeline integration tests

### PR 4

- optional real provider adapter
- prompt fixtures
- UI refinements for fallback state

## Summary Recommendation

The best path for this issue is not to jump straight into calling an SLM. The safest and fastest implementation is:

1. deterministic knowledge cards
2. deterministic retrieval
3. strict summary context
4. provider abstraction with local template provider first
5. verifier and fallback before any real model dependency

That gives the team a grounded, testable summary stack that can later swap in Qwen, SmolLM, MiniCPM, or another provider without changing pipeline contracts.
