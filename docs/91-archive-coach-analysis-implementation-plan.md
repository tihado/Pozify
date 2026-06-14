# Pozify Coach Analysis Expansion Plan

Status: historical implementation plan for the coach-analysis branch work.

Use [../README.md](../README.md) and [01-docs-index.md](01-docs-index.md) for the current runtime
workflow and operational commands.

## Objective

Extend Pozify's grounded coach-summary pipeline with a dataset-ready knowledge layer so the app can incorporate richer exercise expertise without turning the product into a generic chatbot.

The target outcome is:

- Keep summary generation grounded in structured artifacts.
- Expand exercise knowledge through deterministic retrieval.
- Preserve verifier safety rules and current UI/report interfaces.
- Create a clean path for later ingesting open datasets such as `hasaneyldrm/exercises-dataset`.

## Why This Fits The Current Codebase

Pozify already has the right runtime shape:

- `src/pozify/pipeline.py` orchestrates the end-to-end flow.
- `src/pozify/knowledge_cards.py` provides deterministic card retrieval.
- `src/pozify/slm/prompting.py` injects knowledge cards into the SLM prompt.
- `src/pozify/steps/verifier.py` rejects unsafe or ungrounded output.

That means the highest-leverage change is not a chat finetune. It is a stronger knowledge-source layer that can feed better grounded cards into the existing summary step.

## Recommended Data Strategy

### Primary data source

Use `hasaneyldrm/exercises-dataset` as the main source for exercise-grounding expansion because it maps naturally to Pozify's current `knowledge_cards` architecture:

- exercise names
- target muscles
- equipment
- execution cues
- standardized instructions

### Secondary sources

Use conversational datasets such as:

- `onurSakar/GYM-Exercise`
- `HazSylvia/Fitness_Unformatted`
- `chibbss/fitness-chat-prompt-completion-dataset`

only later, and only for:

- coach voice calibration
- response phrasing
- optional supervised examples for tone

These should not be the primary source for grounded issue reasoning.

## Implementation Steps

### Step 1. Plan and branch setup

Deliverables:

- This plan document.
- New branch `ht/coach-analysis`.

Commit:

- `docs: add coach-analysis implementation plan`

### Step 2. Add dataset-ready card pack loading

Goal:

Allow Pozify to load extra knowledge cards from local JSON packs while keeping built-in cards as safe defaults.

Deliverables:

- A local card-pack loader in `src/pozify/knowledge_cards.py` or adjacent module.
- Deterministic merge rules:
  - built-in cards always exist
  - external packs may add new cards
  - external packs may enrich known labels without breaking retrieval order
- Environment-based configuration for optional pack paths.

Why:

This lets us transform open datasets into card packs offline and plug them into the existing summary pipeline without changing the model contract.

Commit:

- `feat: load external coach knowledge card packs`

### Step 3. Add a sample coach-analysis pack

Goal:

Ship one real example pack that demonstrates how open dataset knowledge will be represented inside Pozify.

Deliverables:

- A checked-in JSON pack under `data/knowledge_cards/`.
- Enriched entries for existing exercises such as:
  - squat
  - push-up
  - shoulder press
- Additional factual fields expressed through current card schema:
  - execution cues
  - equipment context
  - target-muscle context

Commit:

- `feat: add sample external coach knowledge pack`

### Step 4. Surface card provenance in artifacts

Goal:

Make it observable when coach summaries used built-in cards versus dataset-backed cards.

Deliverables:

- Final report artifact metadata indicating:
  - whether external packs were loaded
  - which pack files were used
  - how many retrieved cards came from external packs

Commit:

- `feat: expose coach knowledge provenance in artifacts`

### Step 5. Add tests for deterministic retrieval and pack loading

Goal:

Protect the summary pipeline from nondeterministic knowledge changes.

Deliverables:

- Unit tests for:
  - valid external pack parsing
  - deterministic merge behavior
  - duplicate-card override rules
  - retrieval by label with external packs enabled
  - pipeline artifact provenance

Commit:

- `test: cover external coach knowledge retrieval`

### Step 6. Validate end-to-end behavior

Goal:

Confirm the current app still runs and contract tests still pass.

Deliverables:

- Passing targeted tests:
  - `tests.test_coach_summary`
  - `tests.test_pipeline_contracts`
- No regressions in current UI/report flow.

Commit:

- `test: validate coach-analysis pipeline integration`

## Follow-On Work After This Branch

These items are intentionally out of scope for the first implementation pass but become straightforward after the card-pack layer exists:

1. Build a dataset transformer that converts `hasaneyldrm/exercises-dataset` rows into Pozify card packs.
2. Add a small curated evaluation set of artifact-to-summary pairs.
3. Fine-tune a small instruct model on Pozify-native JSON-to-summary examples.
4. Optionally add a separate chat/Q&A mode using the conversational datasets.

## Success Criteria

- Pozify still generates grounded summaries from structured artifacts.
- The app can ingest richer exercise knowledge through deterministic local packs.
- Retrieval remains explainable and testable.
- Verifier and fallback behavior remain unchanged in safety-sensitive cases.
