# STATUS: live board (every machine updates its own row)

Edit your own row when you start or finish something, then commit and push right away. Pull before you start a new task, so two machines never build the same thing.

Last updated: 13 Nov 13:40 by L1.

## GO-AHEAD from Goro (lead), 13:40: "Yes, go ahead"

L2: build `src/labels_clean.py` + `labels_clean.parquet` and the fraud-whole feature view now. Push `labels_clean.parquet`'s builder as code (deterministic) so every machine can rebuild it; L1 rebuilds `labelhist_v1` from it and retrains without `ip_risk_rank`. If you change `src/windows.py`, add a new view name rather than changing `half_s0`/`half_s1`/`full`, so old runs still reproduce.

## READ FIRST (L2, 13:27): `audit/gap_analysis.md`

1. **Nothing has been uploaded yet.** Our 0.4476 is September against noisy labels; the leader's 0.6457 is the public board. Upload `submissions/1113-1259__stack_v1.csv` now to see where we really are.
2. **63% of the fraud labels in train are ordinary legit rows.** Against cleaned September labels the same models score about 0.79. Raw-label September cannot go much above 0.5, so stop judging features on it.
3. **Fraud episodes are whole in the test month; only the legit background is halved.** Pattern rates in test match the `full` view, not `half_s0`. The density A/B verdict (L2) was wrong for fraud-pattern features. Train full, score Sep full: 0.826 on clean labels, against 0.789 for half and half.
4. **`ip_risk_rank` has almost no signal in October** (rank of pattern rows 0.76 in train months, 0.54 in test). Drop it from every config: `drop_features: [ip_risk_rank]`.

L2 is now building `labels_clean.parquet` and the corrected feature view. L1: `labelhist_v1` should be built from clean labels, otherwise it is counting noise.

**L1 check of the above (13:45):** confirmed on L1's own OOF (base + flow + beh1): 63.3% of fraud-labelled rows sit below P = 0.10 with the same 2.3% fraud rate as ordinary rows. A provisional clean file from L1's OOF with L2's rule relabels 36,381 rows (L2: 36,304), clean fraud share 1.31% (L2: 1.36%). The row count also fits "fraud whole, legit halved": 50% of October's legit rows plus 100% of its clean fraud comes to about 260,700; test has 260,582. `src/postprocess.py --labels clean` now fits and scores against `labels_clean.parquet` (and also prints the score against raw labels). Best L1 model (`reg`) against clean Sep: **0.810**; the same decisions against raw Sep: 0.448, unchanged. L1 will rebuild `labelhist_v1` from clean labels as soon as `labels_clean.parquet` is pushed.

## Who is doing what right now

| Machine | Now | Next | Last result |
|---|---|---|---|
| L1 Goro | `labelhist_v1` (entity label history from other months; claimed from L4, the MacBook has not started) | seeds of the best config at lr 0.03 | Best: stack_v1 0.4476, single base + flow + beh1 0.4468 |
| L2 RTX 4060 | Gap analysis done (`audit/gap_analysis.md`); next is `src/labels_clean.py` + the fraud-whole feature view, waiting for the go-ahead | corrected model and submission; `flow_v2` after that | Clean-label September: 0.789 (half view), 0.826 (full view) for base + flow_v1 without ip_risk_rank |
| L3 RTX 3050 | M3 done s0: CatBoost GPU on base1-flow1-beh1 (half_s0, noip, legit_keep 0.2, 6000 rounds): Sep 0.4329 argmax / 0.4361 after postprocess (LGBM twin: 0.4468 — different family for the stack). oof/preds + submission 1113-1800__L3_catboost_base1-flow1-beh1_s0.csv (sha 420f339d6ebe7f4e) are on this laptop; no S3 access from here, so tell me where to send the parquet files. Now building behaviour_v3 (city distance, ATO travel speed) | M6 NN tonight (S5); CatBoost on labels_clean + full view once they land | M3 CatBoost s0: Sep 0.4361 post |
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
| `behaviour_v3` | **L3** (city distance, ATO travel speed) | planned |
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
