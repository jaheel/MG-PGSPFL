# MG-PGSPFL

**MG-PGSPFL** is a federated learning model for event sequence prediction: given a user's visit history across multiple clients, it predicts the event set of the next visit without sharing raw data. The model uses a dual classification head with frequency-weighted aggregation, and is trained with dual-level regularization.

## Repository Structure

```
MG-PGSPFL_release/
├── README.md
├── requirements.txt
├── model/
│   ├── MG-PGSPFL_full.py   # model training + evaluation
│   └── metrics.py          # evaluation metrics
└── data/
    ├── mimic_iv/preprocessed/
    ├── eicu/preprocessed/
    └── instacart/preprocessed/
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0, numpy, pandas

```bash
pip install -r requirements.txt
```

## Input Files

Place the preprocessed tensors of one dataset under `data/<dataset>/preprocessed/` (see `model/MG-PGSPFL_full.py`, constant `DS_NAME`):

| File | Content |
|---|---|
| `sequences.pt` | dict with `matrix`: FloatTensor `(N_steps, N_classes)` multi-hot visit vectors; `index`: `{patient_id: (start_idx, end_idx, center)}` |
| `train_patients.pt` | `{center: [patient_id, ...]}` |
| `test_patients.pt` | `{center: [patient_id, ...]}` |
| `config.json` | `{"n_icd": ..., "centers": [...]}` |

## Run

1. Set `DS_NAME` (in `model/MG-PGSPFL_full.py`) to `mimic_iv`, `eicu`, or `instacart` and adjust hyperparameters at the top of the file if needed (`COMM_ROUNDS`, `GAMMA_MIN`, `MU`, `NU`, `LR`, ...).
2. Run:

```bash
python model/MG-PGSPFL_full.py
```

Results are written to `outputs/<dataset>/MG-PGSPFL_lr..._seed..._<timestamp>/`:

- `final_metrics.json` — final train/test metrics
- `training_history.csv` — per-round metrics
- `final_model.pth` — aggregated global parameters
- `run_config.json` — hyperparameter record
- `training_log.txt` — console log
