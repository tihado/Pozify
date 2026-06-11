# Exercise Router Training Report

Generated after the Modal training run on June 11, 2026.

## Summary

The exercise router was trained and evaluated for `squat`, `push_up`, `shoulder_press`, and
`unknown`. The final active artifact is the scikit-learn baseline because it scored slightly higher
than the BiLSTM temporal model in the artifact selection evaluation.

| Field | Value |
| --- | --- |
| Selected model | `baseline.joblib` |
| Selected artifact | `router.joblib` |
| Selection rule | Highest accuracy, then unknown rejection rate; baseline wins ties |
| Local active path | `models/exercise_router/active/router.joblib` |
| Temporal artifact path | `models/exercise_router/active/temporal.pt` |

## Data

| Metric | Value |
| --- | ---: |
| Feature examples | 134 |
| Window count | 2,224 |
| Failed feature extractions | 0 |
| Push-up windows | 287 |
| Shoulder press windows | 646 |
| Squat windows | 659 |
| Unknown windows | 632 |

Unsupported Riccio dataset classes such as bicep curl are mapped to `unknown`.

## Training Setup

| Model | Setup |
| --- | --- |
| Baseline | scikit-learn `HistGradientBoostingClassifier` over engineered 30-frame window vectors |
| Temporal | PyTorch BiLSTM over 30-frame feature tensors on Modal A10 |

The BiLSTM hyperparameters follow the Riccio exercise-classification paper:

| Hyperparameter | Value |
| --- | ---: |
| Epochs | 73 |
| Hidden units | 73 |
| Dropout | 0.2174 |
| Learning rate | 0.0004 |
| Batch size | 54 |
| Final training loss | 0.0003 |

## Training Metrics

| Model | Validation accuracy | Unknown rejection rate |
| --- | ---: | ---: |
| Baseline | 0.9933 | Not reported in baseline training stage |
| BiLSTM temporal | 0.9820 | 0.9921 |

## Selection Evaluation

The final evaluation scored every available trained artifact on the cached router windows.

| Model | Artifact | Accuracy | Unknown rejection rate |
| --- | --- | ---: | ---: |
| Baseline | `baseline.joblib` | 0.9987 | 0.9984 |
| BiLSTM temporal | `temporal.pt` | 0.9964 | 0.9984 |

### Baseline Precision/Recall

| Label | Precision | Recall |
| --- | ---: | ---: |
| `squat` | 0.9985 | 0.9985 |
| `push_up` | 0.9965 | 1.0000 |
| `shoulder_press` | 0.9985 | 0.9985 |
| `unknown` | 1.0000 | 0.9984 |

### BiLSTM Precision/Recall

| Label | Precision | Recall |
| --- | ---: | ---: |
| `squat` | 0.9954 | 0.9954 |
| `push_up` | 0.9931 | 1.0000 |
| `shoulder_press` | 0.9969 | 0.9938 |
| `unknown` | 0.9984 | 0.9984 |

## Confusion Matrices

Rows are true labels. Columns are predicted labels.

### Baseline

| True \\ Predicted | `squat` | `push_up` | `shoulder_press` | `unknown` |
| --- | ---: | ---: | ---: | ---: |
| `squat` | 658 | 0 | 1 | 0 |
| `push_up` | 0 | 287 | 0 | 0 |
| `shoulder_press` | 0 | 1 | 645 | 0 |
| `unknown` | 1 | 0 | 0 | 631 |

### BiLSTM Temporal

| True \\ Predicted | `squat` | `push_up` | `shoulder_press` | `unknown` |
| --- | ---: | ---: | ---: | ---: |
| `squat` | 656 | 0 | 2 | 1 |
| `push_up` | 0 | 287 | 0 | 0 |
| `shoulder_press` | 3 | 1 | 642 | 0 |
| `unknown` | 0 | 1 | 0 | 631 |

## Reproduction Commands

```bash
uv run modal run scripts/exercise_router_modal.py --stage train-baseline
uv run modal run scripts/exercise_router_modal.py --stage train-temporal
uv run modal run scripts/exercise_router_modal.py --stage evaluate
```

Download the selected artifacts after evaluation:

```bash
uv run modal volume get --force pozify-router-models /router.joblib models/exercise_router/active/router.joblib
uv run modal volume get --force pozify-router-models /temporal.pt models/exercise_router/active/temporal.pt
uv run modal volume get --force pozify-router-models /router_selection.json models/exercise_router/active/router_selection.json
```

## Notes

These metrics are router-window metrics from the current Modal feature cache, not a claim of
generalization to every capture setup. Add more custom `unknown` clips and independent held-out video
sets before treating the router as production-grade.
