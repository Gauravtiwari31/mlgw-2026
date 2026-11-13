# Strategy from 13 Nov evening to the close (15 Nov 10:00 IST)

L2, 13 Nov ~19:30. Built on `strategy.md` section 2 (Amazon ML Challenge lessons) and everything the board has told us.

## Where we stand

| Upload | Public |
|---|---|
| `1113-1401__L2_lgbm_fw-clean_b1-f1-beh12_s0__clean.csv` | **0.4987** (best; rank 11) |
| `1113-1828__L2_fwt3_blend3.csv` (type rules) | 0.4738 (rules withdrawn) |
| probes: qr_swap / mule / velocity / collusion only | 0.1261 / 0.1235 / 0.1109 / 0.1039 |

## What the board has proven

1. October has the same ~2.3% ordinary-looking fraud as train. It cannot be found from this data
   (`audit/leaderboard_analysis.md`). A perfect detector of the visible fraud scores about 0.57-0.59.
2. Our visible-fraud predictions are precise: moving 180 qr_swap predictions cost 0.025. The original
   class types are right; do not relabel types by our own reading of the definitions.
3. Collusion and velocity probes are low because their undetectable rows outnumber their visible rows
   2 to 3 times, and October has fewer bots and no more collusion-shaped sales than September
   (round-amount merchant sales 12.7 vs 13.1 per 10k). There is no large missed pocket to fix.
4. Rare-class F1 on the public part moves about +-0.03 by chance (collusion has ~26 predicted rows there).

## Amazon ML lessons applied

| Amazon lesson | Here |
|---|---|
| Lost on recall; never measured the ceiling | Ceiling measured: about 0.59. We sit at 0.50, simulation says 0.53 |
| Too few experiments | Every change is now scored on Sep with simulated test noise before an upload |
| No stacker | Stacker over OOFs is still worth doing, but expect +0.005, not more |
| Weakest slice found, never fixed | Weakest classes are fixed by the noise, not the model; probes proved it |
| Best public upload becomes the final | Protect the private score: prefer stable, seed-averaged files |

## Plan

1. **Protect the private score (highest value, low risk).** The final is our best public upload,
   scored on the other 70%. A single-seed model is noisier there. Before the close, upload the
   seed-averaged blend of the 0.4987 line (`1113-1414__L2_fwclean_blend4.csv`, 109 rows different)
   and the team stack; whichever public score is highest becomes the final.
2. **Small, testable gains**, each checked on Sep plus simulated noise, uploaded only if the
   simulation gain is above 0.005:
   - more model diversity (CatBoost / XGBoost on the same features) and the stacker
   - sim_swap vs account_takeover separation (both about 0.40 under noise, ceiling about 0.47)
   - fake_cashback (0.42 under noise)
3. **Keep looking for the leader's signal**, but only with hypotheses that can be tested on train
   labels first. Tested and ruled out so far: every column, ids, links, time, running state, row
   order, txn-id hashes, duplicates, entity history, a 416-feature detector (AUC 0.53).
4. **No more label-type rules** without a one-class upload test.

## Upload budget

4 left on 13 Nov, 10 on 14 Nov, 10 on 15 Nov before 10:00.
Keep 2 for the final morning. Never upload without a simulated score next to it in `uploads_log.csv`.

## 13 Nov night: the three Amazon-winner methods, tested (L2)

All scored on Sep with test-like noise against the 0.5019 + stage-2 blend (0.5361):

| Method (Amazon equivalent) | Test-like | Change |
|---|---|---|
| Stacker over 12 run families, XGBoost depth 3 (`src/stack2.py`) | 0.5332 | -0.003 |
| Same, depth 5 | 0.5307 | -0.005 |
| Pair specialists for 6 confused type pairs (cross-encoder idea, `src/pair_specialist.py`) | 0.5330 | -0.003 |
| Sequence transformer over sender histories, 15 epochs (`src/seq_transformer.py`) blended at 0.1-0.3 | 0.5362-0.5365 | +0.000 |

None is used. With near-identical strong inputs, a fitted stacker and specialists overfit the noisy training
types; a plain average of strong models is the better final decision here.
