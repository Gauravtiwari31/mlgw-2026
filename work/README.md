# work/: who does what, on which machine

One folder per machine. Open your folder, read `TASKS.md`, start at the top.
The reasoning behind every task is in `../research.md` (referenced as A1, D2, T4 and so on).

**No pipeline code exists yet.** Ghonchu writes it into `src/` next, once names and laptop owners are confirmed. Until then: do the Setup section of your TASKS.md and read `research.md`.

## 1. Machines

| ID | Machine | GPU | Role | Folder |
|---|---|---|---|---|
| L1 | Goro's laptop | none | Lead: CV harness, base features, uploads, stacker, post-processing | `L1_goro_lead/` |
| L2 | RTX 4060 laptop | 8 GB | Money-flow features, main GBDT line, overnight GNN | `L2_rtx4060_flow_main/` |
| L3 | RTX 3050 laptop | 4–6 GB | Behaviour features, CatBoost, neural net | `L3_rtx3050_behaviour_nn/` |
| L4 | MacBook | none (no CUDA) | Audit and drift, density A/B test, episodes, label cleaning, stage-2 features | `L4_macbook_audit_episodes/` |
| AWS | $250+ credits | on demand | Everything that is only "fit a model": one-vs-rest binaries, feature screen, stage-2 fit, final retrains, reproduce test | `AWS/` |
| Kaggle | free tier, 4 accounts | T4 / P100 | Seeds and config sweeps | `KAGGLE/` |

The MacBook replaces the three CPU laptops in the first plan. It keeps the work that needs a person thinking (audit, episodes, labels). The unattended training jobs moved to AWS.

Assumptions to correct at the merge meeting: the MacBook has at least 16 GB RAM; the 3050 has 4 GB VRAM.

## 2. People (fill in names)

| Person | Machine | Also owns | Name |
|---|---|---|---|
| P1 | L1 | Kaggle dataset mirror, uploads | Goro |
| P2 | L2 | | ______ |
| P3 | L3 | | ______ |
| P4 | L4 MacBook | AWS console and budget | ______ |

Everyone runs their own Kaggle account (see `KAGGLE/TASKS.md`).

## 3. Timeline (IST)

| Slot | Day 1 (13 Nov) |
|---|---|
| S1 11:00–12:30 | Setup, harness, baseline upload |
| S2 12:30–16:00 | v1 features, density A/B test |
| S3 16:00–17:00 | First relational model, upload, error analysis |
| S4 17:00–22:00 | v2 features on the weakest classes, episodes, label cleaning |
| S5 22:00–08:00 | Long runs (GNN, NN, seeds, AWS jobs). Shifts: 2 awake, 2 asleep, 4 hours each |

| Slot | Day 2 (14–15 Nov) |
|---|---|
| S6 08:00–12:00 | Stacker v1, stage-2 features |
| S7 12:00–20:00 | Weakest-slice work, seed averaging |
| S8 20:00–06:00 | Final retrains, clean-clone reproduce test |
| S9 15 Nov 06:00 | Code freeze |
| S10 15 Nov by 09:30 | Repo link submitted (deadline is 10:00) |

## 4. Shared contracts (do not break these)

**Label order.** Every probability file uses these 12 columns in this order:

```
legit, mule_account, velocity_bot, layering_chain, sim_swap, account_takeover,
phishing_collect, qr_swap, refund_abuse, fake_cashback, loan_app_extortion, merchant_collusion
```

**`folds.csv`** is built by `python -m src.folds` on each machine. It is deterministic (hash of `txn_id`), so everyone gets the same file; it must print `sha256=6a366df1a48b8439`. It is not in git and nobody edits it.

| Column | Meaning |
|---|---|
| `txn_id` | every train and test row |
| `split` | `train` or `test` |
| `month_fold` | 0 = Jul, 1 = Aug, 2 = Sep, -1 = Oct-dated train rows and all test rows |
| `is_sep_val` | 1 for Sep train rows |
| `half_s0`, `half_s1` | random 0/1 per train row (seeds 0 and 1); 1 for every test row |

**Density views** (research.md D2). Each relational feature file exists in up to three views:

| View | Same-month neighbours counted | Use |
|---|---|---|
| `half_s0` | rows with the same `half_s0` value | default for training; build this first |
| `half_s1` | rows with the same `half_s1` value | augmentation, second seed |
| `full` | all rows | only if the L4 A/B test says full density is better |

In every view: earlier-month rows always count, later-month rows never count, and test rows see all test rows plus the Oct-dated train rows.

**Feature files:** `features/<group>_v<k>__<view>.parquet`, keyed by `txn_id`, train and test together. Bump `v<k>` on any change; never overwrite a version that a logged run used.

| Group | Owner |
|---|---|
| `base` | L1 |
| `flow` | L2 |
| `behaviour` | L3 |
| `episodes`, `labelhist`, `stage2` | L4 |

**Run outputs:** `oof/<run_id>.parquet` (train rows) and `preds/<run_id>.parquet` (test rows): `txn_id` plus the 12 probability columns. `run_id` = `<machine>_<model>_<features>_<seed>`, for example `L2_lgbm_flow2-beh1_s0`.

**Run log:** `src.train` writes `runs/<run_id>.json` (commit, config, feature versions, Sep macro-F1, 12 per-class F1 values). One file per run, so git never conflicts. Commit it with your config. `runs.csv` at the repo root is only for runs made outside `src.train` (for example the starter notebook).

**Code:** `README.md` at the repo root has setup, the train commands and how to write a feature group with `src/windows.py`.

**Where things live**
- Code: one **private** GitHub repo (sharing code between teams is disqualifying). Opened up for the organisers only when submitting.
- Data, features, OOF, preds: one S3 bucket, same folder names as above. A shared Drive works as a fallback.
- Kaggle: private datasets mirroring `data/`, `features/` and the code.

**Windows and macOS together:** parquet and CSV files are identical on both. Use forward slashes and `pathlib` in code, and keep file names lower-case.

**Uploads** go through P1 only. Each one needs a `runs.csv` line and a row in `uploads_log.csv`.

**Git:** small commits, pull before push, no AI co-author or watermark lines in commit messages.

## 5. Handoff map

| Produces | File | Consumed by |
|---|---|---|
| L1 | `folds.csv`, `features/base_*` | everyone |
| L2 | `features/flow_*` | L3, L4, AWS, Kaggle |
| L3 | `features/behaviour_*` | L2, L4, AWS, Kaggle |
| L4 | `audit/density_ab.md`, `audit/drop_list.txt`, `audit/errors_v*.md` | L1 (decisions), L2 and L3 (what to build next) |
| L4 | `labels_clean.parquet`, `features/episodes_*`, `features/labelhist_*` | all model trainers, L1 (episode smoothing) |
| L4 | `features/stage2_*` | AWS (stage-2 fit) |
| L2, L3, AWS, Kaggle | `oof/*`, `preds/*` | L1 stacker, L4 (cleanlab, stage-2 features) |
| AWS | `audit/time_consistency.csv` | L4 (drop list) |
| AWS | `reproduce_report.md` | L1 (final checklist) |
| L1 | `submissions/*.csv`, `uploads_log.csv` | the leaderboard |

## 6. Rules everyone follows

1. Never use raw `app_version`. Use `ip_risk_score` only as a within-month rank (T2).
2. Never compute a same-month count over all train rows for a training feature (T1).
3. A row's own label, or labels from its own month, never feed its features (T4).
4. No hand-labelling of test rows. Rules written in code are fine (T9).
5. Fixed seeds. Every run has a config file in `configs/` and a commit.
6. Report Sep macro-F1 **and** the 12 per-class F1 values, never the single number alone.
7. An LB gap under 0.01 is noise (T6).
