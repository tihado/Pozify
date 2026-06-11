# Exercise Router Training Report

Generated after the Modal training run on June 11, 2026.

## Summary

The exercise router was trained and evaluated for `squat`, `push_up`, `shoulder_press`, and
`unknown`. The final active artifact is the BiLSTM temporal model. The scikit-learn baseline is
retained as a reference and fallback artifact.

| Field | Value |
| --- | --- |
| Selected model | `temporal.pt` |
| Selected artifact | `temporal.pt` |
| Selection rule | Prefer BiLSTM temporal when available; baseline falls back when temporal is missing |
| Local active path | `models/exercise_router/active/temporal.pt` |
| Baseline artifact path | `models/exercise_router/active/router.joblib` |

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

## Model Complexity

Counts were checked from the local trained artifacts with `uv run --extra train`.

| Model | Count type | Value |
| --- | --- | ---: |
| Baseline | Neural-network-style trainable parameters | 0 |
| Baseline | Input features | 2,574 |
| Baseline | Trees | 800 |
| Baseline | Total tree nodes | 20,778 |
| Baseline | Split nodes | 9,989 |
| Baseline | Leaves | 10,789 |
| Baseline | Approximate learned scalar state | 35,915 |
| BiLSTM temporal | Trainable parameters | 294,924 |
| BiLSTM temporal | Input features per frame | 429 |
| BiLSTM temporal | Hidden units | 73 |
| BiLSTM temporal | Layers | 1 |
| BiLSTM temporal | Output classes | 4 |

The baseline is a tree-based `HistGradientBoostingClassifier`, so it does not have trainable tensor
parameters in the same sense as a neural network. The BiLSTM parameter count includes both LSTM
directions and the linear classification head.

## Training Metrics

| Model | Validation accuracy | Unknown rejection rate |
| --- | ---: | ---: |
| Baseline | 0.9933 | Not reported in baseline training stage |
| BiLSTM temporal | 0.9820 | 0.9921 |

## Selection Evaluation

The final evaluation scored every available trained artifact on the cached router windows.
The baseline scored slightly higher on this cache, but the active router is BiLSTM so routing uses
the temporal pose-window sequence directly.

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

Download the active artifact and selection file after evaluation. Download `router.joblib` too when
you want to keep the baseline for comparison or fallback:

```bash
uv run modal volume get --force pozify-router-models /temporal.pt models/exercise_router/active/temporal.pt
uv run modal volume get --force pozify-router-models /router_selection.json models/exercise_router/active/router_selection.json
uv run modal volume get --force pozify-router-models /router.joblib models/exercise_router/active/router.joblib
```

## Notes

These metrics are router-window metrics from the current Modal feature cache, not a claim of
generalization to every capture setup. Add more custom `unknown` clips and independent held-out video
sets before treating the router as production-grade.
