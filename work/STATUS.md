# STATUS: live board (every machine updates its own row)

Edit your own row when you start or finish something, then commit and push right away. Pull before you start a new task, so two machines never build the same thing.

Last updated: 13 Nov 12:55 by L1.

## Who is doing what right now

| Machine | Now | Next | Last result |
|---|---|---|---|
| L1 Goro | `behaviour_v1` features (the 3050 has not started) | base + flow + behaviour model, then stack | Log-loss early stopping: base + flow 0.4399 after post-processing (was 0.4335) |
| L2 RTX 4060 | (L2: fill in) | `flow_v2`: hop matching, merchant collusion | `L2_lgbm_base1-flow1_s0`: 0.4359 Sep |
| L3 RTX 3050 | Not started | | |
| L4 MacBook | Not started | | |

## Claimed feature groups (claim here before you build one)

| Group | Owner | State |
|---|---|---|
| `base_v1` | L1 | done |
| `flow_v1` | L2 | done (L1 rebuilt it locally from the code; identical) |
| `flow_v2` | L2 | planned |
| `behaviour_v1` | **L1** (moved from L3: the 3050 has not started) | done, 68 features |
| `behaviour_v2` | **L1** | done, 34 features: pair repeats, receiver repeat/evening/round shares, collect fan-in |
| `episodes_v1`, `labelhist_v1` | L4 | open |

If L3 starts: pull first. `behaviour_v1` and `behaviour_v2` are done on L1; take CatBoost (M3) and the neural net (M6) on the existing groups, and `behaviour_v3` (city distance, ATO travel speed) if you want features.

## Best scores so far (honest Sep macro-F1)

| Run | Sep, plain argmax | Sep, after post-processing | Submission |
|---|---|---|---|
| `L1_lgbm_base1_s0` | 0.3851 | 0.3978 | `1113-1201__L1_lgbm_base1_s0.csv` |
| `L1_lgbm_base1abc1_s0` | 0.4222 | 0.4247 | none |
| `L2_lgbm_base1-flow1_s0` | 0.4359 | 0.4335 (L1 rerun) | none |
| **`L1_lgbm_base1-flow1_ll_s0`** (log-loss stopping, lr 0.05) | 0.4387 | **0.4399** | **`1113-1230__L1_lgbm_base1-flow1_ll_s0.csv` (best so far)** |

## Requests between machines

- **L1 → L2:** feature code is deterministic, so L1 rebuilds your groups from the code rather than copying files. Keep pushing code; no need to send parquet files.
- **L1 → L2: add `early_stop_metric: logloss`, `learning_rate: 0.05`, `early_stopping: 100` to your configs.** Same features, +0.006 Sep after post-processing (models now train 58 trees instead of 22). The default in `src/train.py` stays `macro_f1` so your old runs still reproduce.
- **L1 → L2:** don't change `src/train.py` defaults without a note here.
- **L1 → L2, idea for `flow_v2`:** `fake_cashback` is the weakest class (0.27). Beyond the pair reverse edge, count how many *distinct* accounts this row's receiver sent ₹1–₹10 to in the past 24 h / 7 d (a fraudster seeds many victims). Your `acct` role-swap events are the right tool, so it is yours if you want it.
- **Window library fix (12:35):** `window_agg` now re-sorts Polars' rolling output before aligning it; some keys (city names) came back in hash order. Your `flow_v1` used integer keys, which passed the check, so it is unaffected.
