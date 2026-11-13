# Reproduce report

Fresh clone of `main` at `3be9b34` in `E:\mlgw_repro`, data files copied in, `python -m src.reproduce`,
13 Nov 2026 22:00-22:32 IST, RTX 4060 laptop (L2).

Target: `submissions/1113-1912__L2_lgbm4-xgb1_blend5.csv` (public 0.5014, best at the time).

| Step | Result |
|---|---|
| folds, base_v1, labels_clean --from-changes | ok (8 s) |
| flow / behaviour / behaviour2 in fw_s0 and fw_s1 | ok (3.5 min) |
| 4 LightGBM | ok (13.5 min) |
| XGBoost (GPU) | ok (3.2 min) |
| blend | ok |

**Result: 3 of 260,582 rows differ** (phishing -> cashback 1, takeover -> layering 1, sim_swap -> takeover 1).
That is GPU XGBoost run-to-run variation; every step is otherwise deterministic.

`src/reproduce.py` now targets the newer best, `1113-2140__L2_blend8_beh3.csv` (public 0.5019), and adds the
behaviour_v3 and clean label-history builds and the three extra XGBoost seeds.
