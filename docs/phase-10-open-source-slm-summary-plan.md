# Phase 10 Plan: Open-Source SLM Summary Provider

## Objective

Attach a real open-source small language model to the existing summary interface while preserving:

- deterministic context building
- strict verifier checks
- conservative fallback behavior
- existing JSON contracts and UI flow

The summary pipeline must remain:

```text
summary_context -> summary_provider -> verifier -> fallback
```

The SLM must never bypass verification.

## Current State

The current stack already has the right control points:

- deterministic `knowledge_cards` retrieval
- provider abstraction in `summary_provider.py`
- grounded context builder in `summary_context.py`
- verifier and fallback in `pipeline.py` + `coach_summary.py`

What is missing:

- a real model-backed provider
- output parsing/validation for model responses
- provider metadata/debug artifacts
- rollout controls and local run instructions

## Technical Direction

### Initial real provider

Implement one opt-in provider:

- `slm_local`

Back it with one local backend:

- `transformers`

Recommended starter model:

- `Qwen/Qwen2.5-3B-Instruct`

Why:

- good instruction following for structured JSON
- realistic open-source SLM target
- compatible with a local Python integration path

### Safety model

The model only generates a draft. The system still:

1. builds deterministic context
2. requests strict JSON output
3. validates output structure
4. runs verifier checks
5. falls back if anything fails

## Scope

### In scope

- local SLM backend interface
- `transformers` backend implementation
- `slm_local` provider
- provider output parsing and validation
- summary generation metadata artifact
- pipeline/report metadata for provider and fallback use
- tests for provider success/failure/fallback behavior
- README instructions for local SLM usage

### Out of scope

- remote inference APIs
- training or fine-tuning
- changing summary contract shape
- replacing verifier with model judgment

## Environment Variables

- `POZIFY_SUMMARY_PROVIDER=template|mock|unsafe_mock|slm_local`
- `POZIFY_SUMMARY_BACKEND=transformers`
- `POZIFY_SUMMARY_MODEL=Qwen/Qwen2.5-3B-Instruct`
- `POZIFY_SUMMARY_MAX_TOKENS=512`
- `POZIFY_SUMMARY_TEMPERATURE=0.2`
- `POZIFY_SUMMARY_TIMEOUT_SEC=20`

## File-Level Changes

### New files

- `docs/phase-10-open-source-slm-summary-plan.md`
- `src/pozify/steps/summary_slm_backend.py`
- `tests/test_summary_provider.py`

### Updated files

- `src/pozify/steps/summary_provider.py`
- `src/pozify/steps/coach_summary.py`
- `src/pozify/pipeline.py`
- `app.py`
- `web/app.js`
- `README.md`
- `pyproject.toml`
- `tests/test_coach_summary.py`
- `tests/test_pipeline_contracts.py`

## Implementation Checklist

### 1. Add SLM backend abstraction

- [ ] Create `summary_slm_backend.py`
- [ ] Define a backend protocol / result shape
- [ ] Implement `TransformersSummaryBackend`
- [ ] Read model/config from env
- [ ] Return raw model text and backend metadata
- [ ] Fail with explicit runtime errors when backend dependencies are missing

### 2. Add provider output parsing and metadata

- [ ] Define a `SummaryGenerationResult` dataclass
- [ ] Make providers return payload + metadata
- [ ] Add strict JSON extraction/parsing
- [ ] Validate required keys and list/string field types
- [ ] Preserve existing template/mock providers

### 3. Add real SLM provider

- [ ] Implement `OpenSourceSlmProvider`
- [ ] Build prompt from `build_prompt_contract(context)`
- [ ] Force JSON-only output instructions
- [ ] Parse and validate model output
- [ ] Surface parse errors for fallback handling

### 4. Wire provider metadata into pipeline

- [ ] Update `coach_summary` to expose summary draft metadata
- [ ] Update `pipeline.py` to record:
  - provider name
  - backend name
  - model name
  - parse success
  - verification pass/fail
  - fallback used
- [ ] Write `summary_generation.json`
- [ ] Include summary metadata in `final_report.artifacts`

### 5. Keep verifier/fallback mandatory

- [ ] Continue verifying all provider outputs
- [ ] Use fallback on:
  - parse failure
  - malformed JSON
  - invalid contract shape
  - verifier failure
- [ ] Add clear notes for fallback reason

### 6. Surface summary source in app/report

- [ ] Return provider metadata from API
- [ ] Show summary source/fallback state in UI
- [ ] Keep UI conservative when fallback was used

### 7. Add tests

- [ ] Provider returns valid payload with template backend
- [ ] `slm_local` provider can use a mocked backend
- [ ] Parse failures trigger fallback
- [ ] Unsafe model output still fails verifier
- [ ] Final report contains summary metadata
- [ ] Existing contract tests still pass

### 8. Docs and rollout

- [ ] Document optional dependency/runtime needs
- [ ] Document env vars and example commands
- [ ] Keep `template` as default provider

## Suggested Data Shape

### `summary_generation.json`

```json
{
  "provider": "slm_local",
  "backend": "transformers",
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "prompt_contract_version": "v1",
  "parse_ok": true,
  "parse_error": null,
  "verification_passed": true,
  "fallback_used": false
}
```

### `final_report.artifacts` additions

- `summary_generation_path`
- `summary_provider`
- `summary_backend`
- `summary_model`
- `summary_parse_ok`
- `summary_fallback_used`

## Test Matrix

### Provider contract

| Case | Input | Expected |
|---|---|---|
| template provider | normal grounded context | valid payload |
| slm_local provider with mocked backend | valid JSON string | valid payload |
| slm_local malformed text | non-JSON text | parse failure |
| slm_local wrong shape | JSON missing required keys | parse failure |

### Verifier / fallback

| Case | Model output | Expected |
|---|---|---|
| issue hallucination | mentions issue absent from JSON | verifier fails, fallback used |
| variation overcorrection | says variation is wrong | verifier fails, fallback used |
| diagnosis language | uses medical/diagnostic wording | verifier fails, fallback used |
| injury claim | claims prevention | verifier fails, fallback used |
| valid grounded output | aligned with context | verifier passes, no fallback |

### Pipeline integration

| Case | Provider | Expected |
|---|---|---|
| mock/no-video run | template | artifacts still valid |
| real summary path | slm_local mocked backend | `summary_generation.json` written |
| parse failure path | slm_local mocked backend | fallback summary returned |
| verifier failure path | unsafe output | fallback summary returned |

### UI/report checks

| Area | Expected |
|---|---|
| JSON tab | summary metadata present in report artifacts |
| Coach tab | grounded summary still displayed |
| fallback case | confidence/fallback note visible |

## Implementation Order

1. Add plan doc
2. Add backend abstraction + local provider + provider metadata
3. Wire pipeline artifact/report metadata
4. Add tests for provider parsing and fallback
5. Update README usage
6. Run full unit test suite
7. Commit by functionality and push

## Rollout Strategy

### Stage 1

- land backend and provider behind env flag
- keep `template` default

### Stage 2

- compare `template` vs `slm_local` on known clips
- inspect verifier pass/fail rates

### Stage 3

- consider enabling `slm_local` in dev environments only
- keep fallback and template path permanently available

## Acceptance Criteria

- [ ] A real open-source SLM provider can be selected through env config.
- [ ] Provider output is parsed and validated before becoming `CoachSummary`.
- [ ] Verifier still gates all model output.
- [ ] Fallback summary is used on parse or verification failure.
- [ ] Pipeline writes summary generation metadata.
- [ ] UI/report can show whether summary came from template or SLM and whether fallback was used.
- [ ] Full unit test suite passes.
