# STATUS: live board (every machine updates its own row)

Edit your own row when you start or finish something, then commit and push right away. Pull before you start a new task, so two machines never build the same thing.

Last updated: 13 Nov 13:04 by L2.

## Who is doing what right now

| Machine | Now | Next | Last result |
|---|---|---|---|
| L1 Goro | `labelhist_v1` (entity label history from other months; claimed from L4, the MacBook has not started) | seeds of the best config at lr 0.03 | Best: stack_v1 0.4476, single base + flow + beh1 0.4468 |
| L2 RTX 4060 | Idle since 13:02; `flow_v2` not started yet | `flow_v2`: hop matching, merchant collusion, tiny-credit fan-out (L1's idea) | Early-stopping audit (`audit/early_stopping.md`): Sep score is flat from 10 to 500 rounds, no model setting beats seed noise (0.008); lr 0.03 is +0.0046 over 3 seeds |
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
| `labelhist_v1` | **L1** (moved from L4) | building |
| `episodes_v1` | L4 | open |

If L3 starts: pull first. `behaviour_v1` and `behaviour_v2` are done on L1; take CatBoost (M3) and the neural net (M6) on the existing groups, and `behaviour_v3` (city distance, ATO travel speed) if you want features.

## Best scores so far (honest Sep macro-F1)

| Run | Sep, plain argmax | Sep, after post-processing | Submission |
|---|---|---|---|
| `L1_lgbm_base1_s0` | 0.3851 | 0.3978 | `1113-1201__L1_lgbm_base1_s0.csv` |
| `L1_lgbm_base1abc1_s0` | 0.4222 | 0.4247 | none |
| `L2_lgbm_base1-flow1_s0` | 0.4359 | 0.4335 (L1 rerun) | none |
| `L1_lgbm_base1-flow1_ll_s0` (log-loss stopping, lr 0.05) | 0.4387 | 0.4399 | `1113-1230__L1_lgbm_base1-flow1_ll_s0.csv` |
| `L1_lgbm_base1-flow1-beh1_ll_s0` | 0.4457 | 0.4468 | `1113-1246__L1_lgbm_base1-flow1-beh1_ll_s0.csv` |
| `L1_lgbm_base1-flow1-beh12_ll_s0` | 0.4457 | 0.4430 (no gain from v2 on one seed) | `1113-1254__L1_lgbm_base1-flow1-beh12_ll_s0.csv` |
| **`stack_v1`** (LR over the 4 runs above + abc) | 0.4313 (not comparable) | **0.4476** | **`1113-1259__stack_v1.csv` (best so far)** |

**Noise warning:** per-class F1 moves ±0.04 between near-identical runs, so macro-F1 differences under about 0.005 on one seed are noise. Judge a feature group on 2–3 seeds before keeping or dropping it.

## Requests between machines

- **L1 → L2:** feature code is deterministic, so L1 rebuilds your groups from the code rather than copying files. Keep pushing code; no need to send parquet files.
- ~~L1 → L2: use log-loss stopping, +0.006.~~ **Corrected by L2's audit (`audit/early_stopping.md`):** that gain is inside seed noise (0.008). What holds over 3 seeds is lr 0.03 (+0.0046). New configs: `learning_rate: 0.03`; the stopping rule does not matter.
- **L1 → L2:** don't change `src/train.py` defaults without a note here.
- **L1 → L2, idea for `flow_v2`:** `fake_cashback` is the weakest class (0.27). Beyond the pair reverse edge, count how many *distinct* accounts this row's receiver sent ₹1–₹10 to in the past 24 h / 7 d (a fraudster seeds many victims). Your `acct` role-swap events are the right tool, so it is yours if you want it.
- **Window library fix (12:35):** `window_agg` now re-sorts Polars' rolling output before aligning it; some keys (city names) came back in hash order. Your `flow_v1` used integer keys, which passed the check, so it is unaffected.
