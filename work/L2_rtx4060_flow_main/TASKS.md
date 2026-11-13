# L2: Money-flow features + main GBDT line (RTX 4060, 8 GB)

**Owner:** P2 · **Role:** the features that follow the money, the main LightGBM, GPU XGBoost, the overnight GNN.
This is the strongest machine, so the main line lives here.

## Setup

```
nvidia-smi
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Needs: polars, pandas, pyarrow, scikit-learn, lightgbm, xgboost. For the GNN (S5 only): torch with CUDA (take the install command from pytorch.org for your CUDA version) and torch-geometric.
Data: pull `data/`, `folds.csv` and `features/base_v1__full.parquet` from S3.

## Tasks

Build every feature in the `half_s0` view first, then `half_s1`. Windows go **before and after** the row: 1 min, 5 min, 30 min, 1 h, 6 h, 24 h, 7 d. Method: Polars `join_asof`, `rolling` and `group_by_dynamic` on train + test pooled, applying the neighbour rule in `work/README.md` §4.

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S2 | `flow_v1`: receiver fan-in and pass-through (mule) | Per receiver and window: payments in, distinct senders, new_payee share, distinct sender home cities. Then join the receiver's **outgoing** payments (where it is the sender): money out ÷ money in, delay from this payment to its next outflow. Also the sender's inflow just before this payment (mule onward transfers). Receiver account age and KYC | `features/flow_v1__half_s0.parquet` | spot check: mule rows rank high on fan-in in Jul |
| 2 | S2 | `flow_v1`: reverse tiny credit (fake_cashback) | Did the receiver pay this sender ₹1–₹10 in the previous 1 h / 24 h? Time gap and amount ratio | same file | fake_cashback rows light up |
| 3 | S2 | `flow_v1`: QR paid to a P2P receiver (qr_swap) | `txn_type == QR` and `receiver_type == P2P`; qr_vpa_mismatch with null kept; QR fan-in at the receiver per day; receiver account age | same file | pushed to S3 **by 15:00** |
| 4 | S3, **by 16:00** | M1: LightGBM multiclass | base + flow_v1 + behaviour_v1 (from L3). Class weights about inverse-sqrt(frequency). Train Jul–Aug, validate Sep, then month-fold OOF | `oof/` + `preds/` + `runs.csv` line | L1 has it for upload U2 |
| 5 | S4 | `flow_v2`: hop matching (layering_chain) | Next hop: the receiver sends 0.85–1.0x of this amount within 30 min. Same for the previous hop. Chain length forward and backward. 2-hop skip match, because half the links are missing (A1). Hop delay | `features/flow_v2__*.parquet` | layering F1 rises on Sep |
| 6 | S4 | `flow_v2`: merchant collusion | Per merchant: round-number share (amount % 100 or % 500 == 0), sender concentration (top-k share, HHI), repeat senders, onboarding age, daily volume vs its own earlier months | same file | collusion F1 rises on Sep |
| 7 | S4 | M2: XGBoost multiclass on GPU | Same features, `device="cuda"`, `tree_method="hist"` | `oof/` + `preds/` | logged |
| 8 | S5 (night) | M7: GNN edge classifier, **or** XGBoost seeds | PyG GraphSAGE/PNA on the time-windowed graph, edges both directions, edge features = base columns. **If it is not training cleanly by 01:00, drop it and run seeds of M1/M2 instead.** | `oof/` + `preds/`, or seed runs | something useful finishes by 08:00 |
| 9 | S6–S7 | Retrain with L4's outputs | Add `labels_clean`, `episodes`, `labelhist`; then build features for the weakest class in `audit/errors_v*.md` | new runs | each change logged with per-class F1 |
| 10 | S8 | Freeze final configs | Full Jul–Sep fit, fixed seeds; hand the config files to AWS for multi-seed retrains | `configs/final_*.yaml` | AWS confirms they run |

## Handoffs

- **In:** `folds.csv`, `base` (L1); `behaviour` (L3); view decision and drop list (L1/L4); `labels_clean`, `episodes`, `labelhist` (L4).
- **Out:** `features/flow_*` to L3, L4, AWS, Kaggle; OOF and preds to L1 and L4.

## Don'ts

- Don't compute same-month counts over all train rows (T1). Use the half view.
- Don't feed raw receiver, sender or device IDs to the model; use aggregates over them (IEEE lesson).
- Don't use raw `app_version` or raw `ip_risk_score` (T2).
- Don't spend the night debugging the GNN. It is optional; the seeds are not.
