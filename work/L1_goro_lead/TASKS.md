# L1: Lead (Goro's laptop, no GPU)

**Owner:** Goro (P1) · **Role:** CV harness, base features, uploads, post-processing, stacker, final repo.
Nothing heavy runs here. This machine keeps everyone else unblocked.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install polars pandas pyarrow scikit-learn lightgbm
```

Python 3.10 also works if 3.11 is not installed. The five CSV files go in `data/`.

## Tasks

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S1 | Create the private GitHub repo and the S3 bucket (or shared Drive); upload the five data files | Repo layout from `strategy.md` §10. Add all four people. | repo URL and bucket path in the team chat | everyone can pull and download |
| 2 | S1 | Commit `requirements.txt` with pinned versions | `pip freeze` from a clean venv holding every package named in the TASKS files | `requirements.txt` | L2–L4 install from it |
| 3 | S1 | Metric module | Macro-F1 over the fixed label order, plus a per-class precision/recall/F1 table and a confusion matrix (research D1) | `src/metric.py` | all-`legit` scores about 0.08 on Sep |
| 4 | S1, **by 12:30** | Freeze the folds | Columns and rules in `work/README.md` §4 | `folds.csv` | pushed, announced, never edited again |
| 5 | S1 | Base features | hour, weekday, amount features (log, round-number flags, distance to the next 500/1000/2000/5000/10000, amount ÷ balance, amount ÷ month median), account and merchant joins, away-from-home flag, ip_risk as within-month rank. No raw `app_version` (A2–A4) | `features/base_v1__full.parquet` (single-row features, so one view is enough) | file on S3 |
| 6 | S1 | Upload U1: the starter baseline | Run `starter_notebook.ipynb` unchanged; log Sep CV and the public score | `submissions/U1.csv`, first row of `uploads_log.csv` | CV-to-LB gap written down |
| 7 | S1 | Kaggle dataset mirror | Private datasets for `data/`, `features/` and the code (see `KAGGLE/TASKS.md`) | dataset names in the team chat | all four accounts can attach them |
| 8 | S2 | Post-processing module | (a) weighted argmax: 12 weights by coordinate ascent, fitted on Jul–Aug OOF and checked on Sep; (b) prior shift by EM on calibrated probabilities (research D6) | `src/postprocess.py` | improves Sep macro-F1 on the first real OOF |
| 9 | S3 | Upload U2: first relational model from L2 | Apply task 8, then upload | `submissions/U2.csv` | logged |
| 10 | S4 | Decide density view and drift columns | Read L4's `audit/density_ab.md` and `audit/drop_list.txt`; append the decision to the `research.md` change log. Paired upload U3/U4 (ip_risk ranked vs dropped) only if CV cannot separate them | change-log entry | L2 and L3 told which view and columns to use |
| 11 | S6 | Stacker v1 | Multinomial logistic regression on all OOF probability files; compare with a shallow LightGBM stacker and a plain average; month-fold CV | `src/stack.py`, `oof/stack_v1.parquet` | beats the best single model on Sep |
| 12 | S7 | Episode smoothing | Average class probabilities inside each L4 episode before the argmax | updated `src/postprocess.py` | kept only if Sep macro-F1 rises |
| 13 | S7–S8 | Stacker uploads, one fraud-boost probe, seed-averaged final | Keep 2 uploads in reserve | `submissions/`, `uploads_log.csv` | best upload matched to a commit + config |
| 14 | S9–S10 | Final checklist (`strategy.md` §12) | Read `reproduce_report.md` from AWS; submit the repo link from the Submission tab | screenshots saved | **link submitted by 09:30 on 15 Nov** |

## Handoffs

- **Out:** `folds.csv` and `features/base_*` to everyone; decisions to L2 and L3; submissions to the leaderboard.
- **In:** OOF and preds from L2, L3, AWS, Kaggle; audit reports, episodes and clean labels from L4; reproduce report from AWS.

## Don'ts

- Don't upload without a `runs.csv` line and a CV number (T7: the best public upload becomes our final).
- Don't react to LB moves under 0.01 (T6).
- Don't change `folds.csv` after 12:30, for any reason.
- Don't leave the repo link for the last half hour.
