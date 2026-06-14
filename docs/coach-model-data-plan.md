# Pozify Coach Model Data Plan

## Objective

Define a practical data and modeling plan for improving Pozify's coach summary and coach-style guidance using open datasets and the current codebase.

This document answers:

- Which public datasets are actually useful for Pozify's current problem.
- Whether to use RAG, SFT, or both.
- How those choices fit the current pipeline, artifacts, verifier, and UI.
- A concrete implementation roadmap that does not break the current app architecture.

## Current Codebase Fit

Pozify today is not a generic fitness chatbot. It is a structured video-analysis pipeline that ends with a grounded coach summary:

- `video_qc` -> `pose_landmarker` -> `exercise_classifier` -> `rep_analysis` -> `variation` -> `issue_markers` -> `coach_summary` -> `verifier`
- The coach summary step already consumes structured evidence and retrieved knowledge cards.
- The verifier already rejects summaries that invent issues, overcorrect valid variations, or make unsafe claims.

Relevant modules:

- `src/pozify/pipeline.py`
- `src/pozify/steps/coach_summary.py`
- `src/pozify/slm/prompting.py`
- `src/pozify/knowledge_cards.py`
- `src/pozify/steps/verifier.py`

That means any dataset strategy must be evaluated against the real task:

> Generate safe, grounded, structured coaching language from Pose/Rep/Variation/Issue JSON artifacts.

This is **not** the same task as:

- answering open-ended fitness questions,
- free-form nutrition coaching,
- broad lifestyle coaching,
- or generic health chat.

## Recommendation Summary

### Best near-term strategy

Use a **hybrid approach**:

1. **Expand knowledge retrieval first** using a curated exercise knowledge base.
2. **Keep coach summary grounded on structured Pozify artifacts.**
3. **Use lightweight SFT only after you have a Pozify-native training set** built from your own analysis artifacts and target summaries.

### What not to do first

Do **not** directly fine-tune the current coach-summary model on broad fitness chat datasets and expect it to improve the grounded summary step.

That would likely:

- make the model more conversational,
- but also make it more likely to hallucinate,
- talk beyond the provided JSON,
- and fight against the verifier.

### Strong recommendation

For the current codebase, the best order is:

1. **RAG / retrieval upgrade**
2. **Pozify-native supervised dataset creation**
3. **LoRA / SFT on the Pozify-native task**
4. **Optional separate chatbot mode** trained on public chat/Q&A datasets

## Dataset-by-Dataset Assessment

### 1. `onurSakar/GYM-Exercise`

Source notes:

- Hugging Face dataset page shows `1.66k` rows and a single text field formatted in Llama-style instruction format.
- The visible examples include fitness Q&A, but also broader wellness topics such as environmental stressors, work-life balance, mental clarity, and social well-being.

Fit for Pozify:

- Good for **general fitness coach tone**.
- Weak fit for **grounded rep-by-rep coach summary generation**.
- Weak fit for **variation-vs-issue reasoning** tied to structured pose artifacts.

Risks:

- Topic drift beyond movement coaching.
- May encourage generic answer patterns instead of artifact-conditioned reasoning.
- Could increase verifier failures because the model may mention things outside `issue_markers.json`.

Best use in Pozify:

- Use only as **auxiliary style data**.
- Filter to movement-technique and workout-guidance samples.
- Do not use raw as the main SFT dataset for `coach_summary`.

Verdict:

- **Useful later**
- **Not the primary dataset for the current summary pipeline**

### 2. `HazSylvia/Fitness_Unformatted`

Source notes:

- Hugging Face page shows `928` rows.
- Fields appear as `Human` and `Assistant`.
- Examples are cleaner fitness Q&A than the dataset above and closer to gym onboarding and workout guidance.

Fit for Pozify:

- Better than `onurSakar/GYM-Exercise` for conversational coach tone.
- Still not tightly aligned with Pozify's structured JSON-to-summary task.

Risks:

- Still mostly free-form Q&A, not evidence-grounded explanation.
- Lacks explicit inputs like `rep_analysis`, `variation`, `issue_markers`.

Best use in Pozify:

- Use as **secondary SFT data for tone/voice**.
- Convert to JSONL and filter for exercise technique, beginner coaching, and workout-planning responses.
- Consider this dataset for a future `Ask Pozify` chat mode.

Verdict:

- **Good for coach voice**
- **Not enough by itself for the current pipeline**

### 3. `chibbss/fitness-chat-prompt-completion-dataset`

Source notes:

- Hugging Face page shows `245` rows.
- Has `instruction` and `output` fields.
- Content appears to focus on healthy habits, routines, stress, sleep, and lifestyle guidance.

Fit for Pozify:

- Good format for instruction tuning.
- Too small and too broad to be the main fine-tuning source for the current task.

Risks:

- Small size.
- Lifestyle-heavy answers may not improve form-review summaries.
- Could push the model toward broad wellness advice instead of evidence-based movement coaching.

Best use in Pozify:

- Optional augmentation for tone.
- Better suited to a future “habit coach” or “ask a coach” feature than to movement-summary generation.

Verdict:

- **Supplementary only**

### 4. `hasaneyldrm/exercises-dataset`

Based on your description, this is the most strategically valuable dataset for Pozify right now.

Fit for Pozify:

- High fit for retrieval and knowledge expansion.
- It maps well to Pozify's current `knowledge_cards` architecture.
- It improves factual consistency for exercise descriptions, target muscles, equipment, and execution cues.

Why it fits the codebase:

- Pozify already retrieves deterministic cards in `src/pozify/knowledge_cards.py`.
- The prompt builder already injects knowledge cards into the model input.
- The current gap is not “more chat data”; the gap is “richer, broader, more standardized exercise knowledge.”

Best use in Pozify:

- Use this as the basis for a **local knowledge base** first.
- Convert exercise rows into:
  - exercise cards,
  - variation cards,
  - equipment-aware cues,
  - and optional goal-specific cue overlays.
- Later, optionally index it in FAISS/Chroma for semantic retrieval in a separate chat flow.

Verdict:

- **Best immediate dataset to integrate**

### 5. `strova-ai/fitness-tracker-dataset`

Source notes:

- Hugging Face page describes it as a synthetic wearable/activity dataset.
- It contains tabular metrics such as age, gender, height, weight, steps, heart rate, calories, distance, and activity.
- The page also shows dataset-viewer schema issues, which suggests ingestion may need cleanup.

Fit for Pozify:

- Useful for personalization experiments.
- Weak fit for current video-form summary generation.

Why it does not fit immediately:

- Pozify's current app does not reason from wearable timeseries.
- The current `UserProfile` contract is small and categorical.
- There is no current component that transforms tabular physiology into grounded motion advice.

Best use in Pozify:

- Optional later-stage personalization module.
- Useful for future “today's activity context” or “session recommendation” features.
- Not a priority for the current coach summary pipeline.

Verdict:

- **Future personalization dataset, not Phase 1**

## What This Means For Pozify

### The current task is not generic chat SFT

Pozify's core output is:

- bounded by `exercise_classification.json`
- bounded by `rep_analysis.json`
- bounded by `variation.json`
- bounded by `issue_markers.json`
- conditioned by knowledge cards
- checked by a verifier

So the most valuable training examples are not public generic fitness Q&A.

The most valuable examples look like this:

```json
{
  "analysis_json": {
    "user_profile": {},
    "exercise_classification": {},
    "rep_analysis": {},
    "variation": {},
    "issue_markers": {}
  },
  "retrieved_knowledge_cards": [],
  "ideal_coach_summary": {
    "summary": "",
    "what_you_did": [],
    "what_looked_good": [],
    "what_changed_across_reps": [],
    "valid_variation_vs_issue": [],
    "top_fixes": [],
    "next_session_plan": [],
    "confidence_notes": []
  }
}
```

This is already much closer to the current runtime prompt contract than any public chat dataset you listed.

## Recommended Strategy

## Option A: Retrieval-first upgrade

### Why this is the best first move

It fits the current architecture with the least risk.

It improves:

- factual exercise guidance,
- coaching cue coverage,
- equipment-specific instructions,
- and future exercise expansion,

without retraining the model first.

### How to use `hasaneyldrm/exercises-dataset`

Convert exercise rows into a normalized internal artifact such as:

```json
{
  "card_id": "exercise:barbell_bench_press",
  "card_type": "exercise",
  "labels": ["barbell_bench_press", "bench_press"],
  "title": "Barbell Bench Press",
  "summary": "Compound upper-body press.",
  "target_muscles": ["chest"],
  "secondary_muscles": ["triceps", "front_delts"],
  "equipment": ["barbell", "bench"],
  "instructions": [
    "Set the shoulder blades before unracking.",
    "Lower with control to a repeatable touch point.",
    "Press while keeping wrists stacked."
  ]
}
```

### Implementation in current codebase

1. Add a data ingestion script under `scripts/`:
   - `scripts/build_exercise_knowledge_base.py`
2. Store normalized cards under:
   - `data/knowledge/exercises.json`
3. Update `src/pozify/knowledge_cards.py` to:
   - load built-in cards,
   - merge external exercise cards,
   - keep deterministic retrieval for current exercise/variation/issue labels.
4. Keep the current coach-summary flow intact.

### Benefits

- Immediate product improvement.
- No retraining required.
- Low safety risk.
- Reuses the current prompt architecture.

## Option B: Pozify-native SFT for coach summary

### Why this is the right SFT target

If you do SFT, train on the task Pozify actually runs:

- structured evidence in
- structured grounded summary out

### Recommended data source for SFT

Build your own dataset from:

- current artifact pipeline outputs in `runs/`
- synthetic or hand-authored expert summaries
- optional edits by the team

Then optionally augment with:

- filtered examples from `HazSylvia/Fitness_Unformatted`
- filtered examples from `onurSakar/GYM-Exercise`
- filtered examples from `chibbss/fitness-chat-prompt-completion-dataset`

But use public datasets only for:

- tone,
- phrasing,
- concise coaching style,
- and generic structure.

Do not use them to replace the grounded target format.

### SFT training objective

Input:

- current prompt evidence object
- retrieved cards

Output:

- the exact `CoachSummary` JSON contract

### Recommended model

Use the current runtime-aligned model family first:

- `Qwen/Qwen3-14B`

Because the app already uses this model family successfully through Hugging Face Inference.

### Recommended training tools

- TRL if you want closer Hugging Face-native control
- Unsloth if you want fast LoRA iteration on a smaller GPU budget
- Axolotl if you want a cleaner config-driven fine-tuning pipeline

### Best fine-tuning style

- LoRA / QLoRA
- JSON-only target format
- deterministic eval checks against verifier outcomes

### Why not train directly on public chat sets first

Because the task mismatch is too large:

- public chat sets teach “fitness assistant”
- Pozify needs “artifact-grounded explanation generator”

## Option C: Separate chat / Ask-a-Coach feature

This is where the public datasets are actually strongest.

### Best datasets for that feature

- `HazSylvia/Fitness_Unformatted`
- `onurSakar/GYM-Exercise`
- `chibbss/fitness-chat-prompt-completion-dataset`

### Suitable user stories

- “I am new to the gym, where do I start?”
- “Can you give me a 3-day weekly plan?”
- “What should I eat after a workout?”
- “How do I stay consistent?”

### Why keep this separate from coach summary

Because:

- the current summary step must stay grounded
- broad wellness chat should not weaken form-review safety
- the verifier logic is built for artifact-based outputs, not broad coaching conversations

### Suggested architecture

- Keep `coach_summary` as today, evidence-first.
- Add a future `ask_coach` endpoint and UI tab.
- Back that feature with:
  - a filtered fitness Q&A dataset,
  - optional RAG over exercise knowledge,
  - and a different safety policy.

## Recommended Final Architecture

### Phase 1

- Retrieval-only upgrade from exercise knowledge base
- No SFT required

### Phase 2

- Build Pozify-native summary dataset
- Fine-tune a LoRA adapter for structured summary generation

### Phase 3

- Add optional `Ask Pozify` chat mode
- Use public Q&A datasets there

### Phase 4

- Add personalization features using wearable/tabular data
- Use `strova-ai/fitness-tracker-dataset` only if product scope expands into recommendations

## Concrete Implementation Plan

## Step 1: Add external exercise knowledge ingestion

Goal:

- turn exercise reference data into Pozify-ready knowledge cards

Work:

1. Create `data/knowledge/` directory.
2. Add `scripts/build_exercise_knowledge_base.py`.
3. Normalize raw exercise dataset fields into a local JSON schema.
4. Add tests for normalization and card loading.
5. Update `src/pozify/knowledge_cards.py` to load external cards.

Deliverables:

- `data/knowledge/exercises.json`
- updated knowledge card loader
- tests

## Step 2: Expand current retrieval interface

Goal:

- use richer exercise and equipment context without changing app UX

Work:

1. Keep deterministic retrieval for current labels.
2. Add equipment-aware retrieval using `UserProfile.equipment`.
3. Add goal-aware retrieval overlays.
4. Add per-exercise cue prioritization for the summary prompt.

Deliverables:

- richer `retrieve_cards(...)`
- no changes needed in the UI contract

## Step 3: Create a Pozify-native SFT dataset builder

Goal:

- create training examples from the actual runtime task

Work:

1. Add `scripts/build_coach_summary_sft_dataset.py`.
2. Read from:
   - `runs/*/exercise_classification.json`
   - `runs/*/rep_analysis.json`
   - `runs/*/variation.json`
   - `runs/*/issue_markers.json`
   - retrieved knowledge cards
3. Export JSONL rows like:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...structured evidence..."},
    {"role": "assistant", "content": "{...coach summary json...}"}
  ]
}
```

4. Start with hand-authored gold summaries for 100-300 samples.
5. Add validation scripts that reject malformed outputs.

Deliverables:

- `data/sft/coach_summary_train.jsonl`
- `data/sft/coach_summary_eval.jsonl`

## Step 4: Use public datasets only as auxiliary style data

Goal:

- improve coach tone without damaging grounded reasoning

Work:

1. Add `scripts/prepare_public_fitness_chat_data.py`.
2. Ingest:
   - `onurSakar/GYM-Exercise`
   - `HazSylvia/Fitness_Unformatted`
   - `chibbss/fitness-chat-prompt-completion-dataset`
3. Filter out:
   - broad medical claims
   - unrelated wellness topics
   - environment/toxin/social-life content
4. Convert only the most relevant rows into a secondary style corpus.
5. Use this corpus with lower sampling weight than Pozify-native data.

Deliverables:

- `data/sft/public_fitness_style.jsonl`

## Step 5: Fine-tune with LoRA

Goal:

- make the model better at emitting Pozify's exact summary contract

Work:

1. Choose one training stack:
   - TRL
   - Unsloth
   - Axolotl
2. Fine-tune `Qwen/Qwen3-14B`.
3. Use:
   - low temperature at inference
   - JSON-only targets
   - training/eval split with verifier-based metrics
4. Track:
   - JSON validity rate
   - verifier pass rate
   - unsupported-issue mention rate
   - variation-overcorrection rate

Deliverables:

- LoRA adapter
- eval report
- deployment notes

## Step 6: Add pluggable model loading for local or Hub adapters

Goal:

- swap between base model and fine-tuned adapter cleanly

Work:

1. Extend `src/pozify/slm/providers.py`.
2. Add env-driven model selection:
   - base model
   - fine-tuned adapter
   - fallback model
3. Surface model metadata in `final_report.artifacts`.

Deliverables:

- deployable adapter path
- clear model/version tracking in UI and artifacts

## Step 7: Optional future RAG for free-form chat

Goal:

- add a broader coach chat feature without weakening grounded summaries

Work:

1. Add `src/pozify/chat/` module.
2. Index exercise knowledge into:
   - FAISS first, or
   - ChromaDB if persistence/filtering is preferred
3. Use RAG only for:
   - exercise explanations
   - equipment substitutions
   - general programming suggestions
4. Keep this separate from `coach_summary`.

Deliverables:

- `Ask Pozify` feature
- isolated inference path

## Decision Table

| Dataset | Best use in Pozify | Use now? | Notes |
|---|---|---:|---|
| `onurSakar/GYM-Exercise` | auxiliary style SFT | No | useful tone, but too broad for summary task |
| `HazSylvia/Fitness_Unformatted` | auxiliary style SFT / future chat | Later | cleaner coach Q&A, still task-mismatched |
| `chibbss/fitness-chat-prompt-completion-dataset` | auxiliary style SFT / future chat | Later | small, useful for tone only |
| `hasaneyldrm/exercises-dataset` | retrieval knowledge base | Yes | best immediate fit for current architecture |
| `strova-ai/fitness-tracker-dataset` | future personalization | No | not aligned with current app contracts |

## Final Recommendation

If the goal is to improve the current Pozify product, do this:

1. **Integrate `hasaneyldrm/exercises-dataset` as structured knowledge cards first.**
2. **Build a Pozify-native coach-summary SFT dataset from real artifacts.**
3. **Fine-tune Qwen with LoRA on the Pozify-native task.**
4. **Use public fitness chat datasets only as secondary style data.**
5. **Use RAG later for a separate chat feature, not to replace the current grounded summary path.**

This path gives the best balance of:

- safety,
- architecture fit,
- product usefulness,
- and implementation risk.

## Sources

- Hugging Face dataset page for `onurSakar/GYM-Exercise`: https://huggingface.co/datasets/onurSakar/GYM-Exercise
- Hugging Face dataset page for `HazSylvia/Fitness_Unformatted`: https://huggingface.co/datasets/HazSylvia/Fitness_Unformatted
- Hugging Face dataset page for `chibbss/fitness-chat-prompt-completion-dataset`: https://huggingface.co/datasets/chibbss/fitness-chat-prompt-completion-dataset
- Hugging Face dataset page for `strova-ai/fitness-tracker-dataset`: https://huggingface.co/datasets/strova-ai/fitness-tracker-dataset
- Pozify runtime code:
  - `src/pozify/pipeline.py`
  - `src/pozify/steps/coach_summary.py`
  - `src/pozify/slm/prompting.py`
  - `src/pozify/knowledge_cards.py`
  - `src/pozify/steps/verifier.py`
