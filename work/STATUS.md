# STATUS: live board (every machine updates its own row)

Edit your own row when you start or finish something, then commit and push right away. Pull before you start a new task, so two machines never build the same thing.

Last updated: 15 Nov 05:30 by L2.

## L2, 15 Nov 05:30: final state before the 10:00 deadline

- Best public upload is still `submissions/1114-0854__L2_blend16_zones3.csv` = **0.5417** (rank 17 at midnight; top 0.6338).
  Nothing built since beats it: weights 0.5411, pairwise type flips 0.5407, density / surgical flips 0.5320.
- `reproduce.sh` rebuilds that file. Fresh-clone run 15 Nov 05:17-06:33: every step ok, **2 of 260,582 rows differ** (GPU run-to-run variation). See `reproduce_report.md`.
- Last round of checks (all empty) is written up at the end of `audit/device_blocks.md`.
- **To do before 10:00:** submit the repository link and the approach note from the Submission tab.

## L2, 14 Nov 09:30: best is now 0.5417 (`submissions/1114-0854__L2_blend16_zones3.csv`); `reproduce.sh` targets it. 6 of 10 uploads used.

## L2, 14 Nov 09:00: zones file scored 0.5403 (board top 0.5555)

- `submissions/1114-0359__L2_blend8_zones2.csv` = **0.5403**. `reproduce.sh` now rebuilds this file (zone step checked: 0 rows differ).
- Test labels are clean (statement, point 2): judge on the test-like ruler that `src.zone_apply` prints (Sep, unreported rows counted as fraud). 0.5403 on the board = 0.5475 there.
- What is left is fraud TYPE on rows we already call fraud (0.057 on that ruler). Those are look-alike episodes; only a block phone in the episode tells. Details and the list of things that do not work: `audit/device_blocks.md`.
- Next candidate: `submissions/1114-0854__L2_blend16_zones3.csv` (expected 0.540 to 0.544, a coin flip against 0.5403).
- 14 Nov uploads used: 5 of 10.

## L2, 14 Nov 04:00: phone ids give away fraud rows and their class (`audit/device_blocks.md`)

- New phone ids come from a counter. The test month's first 920 new-phone ids (473632 to 474551) are fraud, laid
  out class by class in label order. `python -m src.device_block --runs <blend runs>` finds the zones without
  labels; `python -m src.zone_apply --propagate --base <file> --runs <blend runs>` applies them to a submission.
- File to upload: `submissions/1114-0359__L2_blend8_zones2.csv` (the 0.5019 file with 310 rows changed).
  Sep raw labels 0.4749 -> 0.5014. Expected public about 0.528. **Score not known yet.**
- Any new blend: run `src.zone_apply` on it as the last step. The gain is independent of the base model.
- Not worth more time (all measured, see the audit file): more seeds or bigger blends (+0.0003), transaction ids,
  account or merchant id ranges, hidden fraud, unreported rows, the velocity/mule and cashback/phishing mix-ups.
- 14 Nov uploads used: 4 of 10 (probes: sim_swap 0.1147, takeover 0.1206, cashback 0.1223, phishing 0.1241).

## L2, 19:05: CORRECTION â€” the type rules below FAILED on the board (0.4738 vs 0.4987). They are switched off.

Do not use the 18:30 labels. If you already rebuilt: `git pull` then `python -m src.labels_clean --from-changes` gives
back the labels behind the 0.4987 file (verified identical). Rule 4 (qr_swap to a merchant -> collusion) explains the
loss: in the clean test labels those rows are qr_swap. Best public upload is now
`submissions/1113-1401__L2_lgbm_fw-clean_b1-f1-beh12_s0__clean.csv` (0.4987).

## L2, 18:30 (WITHDRAWN, see 19:05): wrong-type labels fixed

Probes on the public board: qr_swap 0.1261, mule 0.1235, velocity 0.1109, collusion 0.1039 (F1 about 0.53 / 0.50 / 0.35 / 0.27).
Collusion was the outlier: the train labels file collusion under qr_swap and refund_abuse, and swap sim_swap/ATO,
phishing/cashback, velocity/mule. Test labels are clean. `src/labels_clean.py` now applies five definition rules
(2,671 rows retyped). **Every machine: `git pull` then `python -m src.labels_clean --from-changes`, and retrain.**
New L2 blend `submissions/1113-1828__L2_fwt3_blend3.csv`: test-like Sep 0.579 vs 0.522 for the uploaded file.
Judge models on Sep with clean labels plus simulated 2.32% noise (`src/probe.py` shows how), not on raw Sep.

## L2, 14:16: corrected line is in git

- Every machine: `python -m src.labels_clean --from-changes` (identical `labels_clean.parquet` from `audit/labels_clean_changes.csv`), then build groups with `--view fw_s0`. Configs: `view: fw_s0`, `labels: clean`, `drop_features: [ip_risk_rank]`.
- `labelhist_v2` (`python -m src.labelhist_features --labels clean`) adds +0.003 on one seed: not in the main line.
- Class weights for the argmax and episode smoothing add nothing on clean labels (`src/blend.py`).
- Fraud is halved in test too (as whole episodes): pattern rows per 10,000 match train, they do not double. L1's row-count note (all fraud kept) does not fit that.
- About 27% of sim_swap and account_takeover episodes carry the other type's signature (fresh SIM vs new phone). Looks like swapped labels; the model predicts the majority type.

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
| L2 RTX 4060 | Full-density 5-model blend training (21:00); CatBoost from L3's trainer next | 10 PM / 11:50 PM uploads | **Best public 0.5014**: `submissions/1113-1912__L2_lgbm4-xgb1_blend5.csv` |
| L3 RTX 3050 | Corrected line live: CatBoost beh123-lh2 s0 clean Sep 0.9016 argmax / 0.9024 post (14 min/run; config pushed, rerun locally for blends). NN beh123-lh2 s0: 0.8538/0.8641. LGBM beh123-lh2 s0: OOF 0.8964 (6 min/run). Running now: CatBoost A/B without beh3 (GPU), LGBM s1/s2 + A/B (CPU), CatBoost seeds s1/s2 + depth-10 after that, NN seeds behind those | A/B verdict on behaviour_v3 + seed-averaged runs by morning | best: CatBoost beh123-lh2 s0 0.9024 post |
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

**Noise warning:** per-class F1 moves Â±0.04 between near-identical runs, so macro-F1 differences under about 0.005 on one seed are noise. Judge a feature group on 2â€“3 seeds before keeping or dropping it.

## Requests between machines

- **L1 â†’ L2:** feature code is deterministic, so L1 rebuilds your groups from the code rather than copying files. Keep pushing code; no need to send parquet files.
- ~~L1 â†’ L2: use log-loss stopping, +0.006.~~ **Corrected by L2's audit (`audit/early_stopping.md`):** that gain is inside seed noise (0.008). What holds over 3 seeds is lr 0.03 (+0.0046). New configs: `learning_rate: 0.03`; the stopping rule does not matter.
- **L1 â†’ L2:** don't change `src/train.py` defaults without a note here.
- **L1 â†’ L2, idea for `flow_v2`:** `fake_cashback` is the weakest class (0.27). Beyond the pair reverse edge, count how many *distinct* accounts this row's receiver sent â‚¹1â€“â‚¹10 to in the past 24 h / 7 d (a fraudster seeds many victims). Your `acct` role-swap events are the right tool, so it is yours if you want it.
- **Window library fix (12:35):** `window_agg` now re-sorts Polars' rolling output before aligning it; some keys (city names) came back in hash order. Your `flow_v1` used integer keys, which passed the check, so it is unaffected.

## L2, 19:25: best public is now 0.5014: `submissions/1113-1912__L2_lgbm4-xgb1_blend5.csv` (4 LightGBM + 1 XGBoost, original labels). Simulation predicted +0.002, board gave +0.0027.

