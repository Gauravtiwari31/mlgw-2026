# L3: Behaviour features + CatBoost + neural net (RTX 3050, 4–6 GB)

**Owner:** P3 · **Role:** the features about how a sender and a phone behave, CatBoost on GPU, a small neural net overnight.

## Setup

```
nvidia-smi
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Needs: polars, pandas, pyarrow, scikit-learn, lightgbm, catboost. For the neural net (S5 only): torch with CUDA (install command from pytorch.org).
Data: pull `data/`, `folds.csv` and `features/base_v1__full.parquet` from S3.

## Tasks

Build every feature in the `half_s0` view first, then `half_s1`. Windows go **before and after** the row: 1 min, 5 min, 30 min, 1 h, 6 h, 24 h, 7 d. Same Polars method and neighbour rule as L2 (`work/README.md` §4).

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S2 | `behaviour_v1`: bursts (velocity_bot) | Per device and per sender: payment counts in 1/5/15-min windows, inter-arrival mean and std (scripts are regular), distinct receivers, share of gaming/e-commerce, std of login_to_txn_sec. Accounts per device over the whole data (A6) | `features/behaviour_v1__half_s0.parquet` | farm devices stand out |
| 2 | S2 | `behaviour_v1`: SIM swap and takeover | SIM changed since the last payment: `sim_age_hours` < hours since the sender's previous payment. New device: `device_age_days` small, or a device never seen for this sender. `device_city != home_city`. Night hour. amount ÷ balance. Count of payments in the hour after the change. pin_retries | same file | sim_swap and ATO rows separate from legit |
| 3 | S2 | `behaviour_v1`: refunds (refund_abuse) | Per sender: `REFUND` count over ±7 d and ±30 d, refunds ÷ purchases, refund amount equal to an earlier purchase at the same merchant, e-commerce share | same file | pushed to S3 **by 15:00** so L2 can train M1 |
| 4 | S4 | `behaviour_v2`: loan-app extortion | Sender–receiver pair repeat count over 7 d and 30 d, share in hours 20–23, `lending` category, number of distinct senders that repeat-pay the same receiver | `features/behaviour_v2__*.parquet` | loan_app F1 rises on Sep |
| 5 | S4 | `behaviour_v2`: phishing collect | `COLLECT` type, just-under-round flags from `base`, collect fan-in at the receiver across distinct senders, new P2P receiver | same file | phishing F1 rises on Sep |
| 6 | S4 | `behaviour_v2`: city distance | Small lookup table of latitude/longitude for the cities in `accounts.csv`, written in code; distance between `device_city` and `home_city` | same file + `data/city_coords.csv` | ATO F1 rises on Sep |
| 7 | S4 | M3: CatBoost multiclass on GPU | Same features as M1, `task_type="GPU"`, class weights. If VRAM runs out: lower `border_count`, then CPU, then move it to a Kaggle T4 | `oof/` + `preds/` + `runs.csv` line | logged |
| 8 | S5 (night) | M6: MLP or TabM | Standardised numeric features, embeddings for categoricals, weighted cross-entropy, month-fold OOF | `oof/` + `preds/` | finishes by 08:00 |
| 9 | S6–S7 | Retrain with L4's outputs | Add `labels_clean`, `episodes`, `labelhist`; then build features for the weakest behaviour class in `audit/errors_v*.md` | new runs | each change logged with per-class F1 |
| 10 | S8 | Freeze final configs | Full Jul–Sep fit, fixed seeds; hand to AWS and Kaggle for seeds | `configs/final_*.yaml` | confirmed to run |

## Handoffs

- **In:** `folds.csv`, `base` (L1); `flow` (L2); view decision and drop list (L1/L4); `labels_clean`, `episodes`, `labelhist` (L4).
- **Out:** `features/behaviour_*` to L2, L4, AWS, Kaggle; OOF and preds to L1 and L4.

## Don'ts

- Don't compute same-month counts over all train rows (T1). Use the half view.
- Don't use raw `app_version` or raw `ip_risk_score` (T2).
- Don't fight the 4 GB VRAM for long. CatBoost on CPU or on Kaggle is fine.
- Don't type labels or per-row fixes by hand. The city table is public reference data, which is allowed; row labels are not (T9).
