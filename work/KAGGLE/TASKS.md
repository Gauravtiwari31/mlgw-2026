# KAGGLE: free-tier seeds and sweeps (4 accounts)

**Owner:** each person runs their own account. Goro (L1) keeps the datasets current. · **Role:** free GPU and CPU hours for seed averaging and config sweeps.

**Limits to plan around:** about 30 GPU hours per week per account, 12-hour sessions, 2×T4 or one P100, phone verification needed for GPU and internet.

## Setup (S1)

1. Phone-verify every account today. Without it there is no GPU.
2. Goro creates three **private** datasets and shares them with the other three accounts:

| Dataset | Holds | New version when |
|---|---|---|
| `mlgw-data` | the five CSV files + `folds.csv` | once |
| `mlgw-features` | everything in `features/` | each feature release (v1, v2, ...) |
| `mlgw-code` | a zip of the repo at a named commit | each time a config needs newer code |

Shipping the code as a dataset means no GitHub token ever goes into a notebook.

3. One notebook template: attach the three datasets, unzip the code, `pip install` anything missing, run one config, write outputs to `/kaggle/working/`.

```
python -m src.train --config configs/<name>.yaml
```

## Who runs what

| Account | Model family | From |
|---|---|---|
| Goro | LightGBM multiclass seeds (CPU notebook) | L2's M1 config |
| P2 | XGBoost GPU seeds | L2's M2 config |
| P3 | CatBoost GPU seeds (and the full M3 if the 3050 runs out of VRAM) | L3's M3 config |
| P4 | MLP/TabM seeds, or one-vs-rest overflow | L3's M6 config, AWS's M4 configs |

## Tasks

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S1 | Verify accounts, create and share the datasets | Setup steps above | dataset names in the team chat | all four accounts can attach them |
| 2 | S3 | Smoke test | Each account runs the starter-level config end to end | one tiny OOF file each | the template works everywhere |
| 3 | S4 → S5 | First seed batch | 2–3 seeds of your family on the v1/v2 features, started before sleeping | `oof/` + `preds/` files | downloaded and pushed to S3 by the morning |
| 4 | S7 | Second seed batch on final features | Clean labels, `episodes`, `labelhist` included | `oof/` + `preds/` files | pushed to S3 |
| 5 | S8 | Final seeds | Frozen configs, full Jul–Sep fit | `preds/final_*` | pushed to S3 **before 04:00 on 15 Nov** |

After every run: download the outputs, push them to S3 under the shared names, and add the `runs.csv` line (machine = `KG_<account>`).

## Handoffs

- **In:** datasets from Goro; configs from L2 and L3.
- **Out:** `oof/*` and `preds/*` to S3 for L1's stacker.

## Don'ts

- Don't make any dataset or notebook public. Sharing code or predictions between teams is disqualifying.
- Don't burn GPU hours on feature building. Features come from the laptops and AWS; Kaggle only fits models.
- Don't start a run longer than about 10 hours; the session dies at 12.
- Don't count a Kaggle run as reproducible unless its commit and config are logged. The final `reproduce.sh` must not depend on a Kaggle session.
