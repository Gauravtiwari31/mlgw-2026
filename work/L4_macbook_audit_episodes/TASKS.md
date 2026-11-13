# L4: Audit, drift, episodes, labels (MacBook, no CUDA)

**Owner:** P4 (also runs the AWS console, see `../AWS/TASKS.md`) · **Role:** find where the score is lost, then build the episode and label layer that every model uses.

This one machine replaces three laptops from the first plan. It keeps the work that needs judgement. Anything that is only "fit a model and wait" is launched on AWS from here.

## Setup

```
brew install libomp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Needs: polars, pandas, pyarrow, scikit-learn, lightgbm, scipy, cleanlab. `libomp` is what LightGBM needs on macOS.
Data: pull `data/`, `folds.csv` and `features/base_v1__full.parquet` from S3.

## Tasks, in priority order

If time runs short, tasks 1, 2, 4 and 5 matter most. Drop from the bottom.

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S1 | AWS first steps | Billing alarms and the GPU quota request (`../AWS/TASKS.md` tasks 1–2). Ten minutes, do it before anything else | alarms live | confirmed in the team chat |
| 2 | S2, **by 14:30** | **Density A/B test** (research A1, D2) | Don't wait for L2/L3. Build six quick count features yourself (receiver: payments and distinct senders in ±1 h and ±24 h; device: payments in ±5 min; sender: payments in ±1 h) in the `full` and `half_s0` views. Train LightGBM on Jul–Aug twice, once per view. Score both on Sep built at `half_s0` density | `audit/density_ab.md`: two macro-F1 numbers, per-class table, one-line verdict | L1 announces which view is the default |
| 3 | S2 | Sampling and label-shape questions (research Part A, open questions) | Is the test half a random row sample or whole episodes? Compare burst-size distributions Sep vs Oct. Why is the median 1 row per mule receiver and per velocity device? | appended to `audit/density_ab.md` | three questions answered with numbers |
| 4 | S2–S3 | Adversarial validation | LightGBM classifier: Sep rows vs Oct rows, on `base` and then on each feature group as it lands. List the top drifting features | `audit/drop_list.txt` (drop / rank-transform / keep, one line each) | sent to L1, L2, L3 |
| 5 | S3–S4 | Episodes v1 | Edges between rows that share a receiver, a sender or a device within a time window (start with 30 min, try 6 h and 24 h); connected components with `scipy.sparse.csgraph.connected_components`. Cap or split giant components. Per row: episode id, size, duration, distinct senders/receivers/devices | `features/episodes_v1__half_s0.parquet` | fraud rows of one burst share an id far more often than chance |
| 6 | S3 | Error analysis v1 | On L2's first OOF: per-class F1, confusion matrix, worst class, which fraud types get mixed up. Name the next feature to build | `audit/errors_v1.md` | L2 and L3 know their S4 target |
| 7 | S4 | Label history | Per receiver, sender, device: fraud rate and label mix from **other months only** (for test rows: all of Jul–Sep plus the Oct-dated train rows, A5) | `features/labelhist_v1__half_s0.parquet` | leakage check: a model on these features alone scores no higher on its own fold than on Sep |
| 8 | S4–S5 | Label cleaning | (a) Episode-majority relabel: a `legit` row inside an episode dominated by one fraud type. (b) cleanlab `find_label_issues` on the best OOF probabilities. Drop flagged rows; do not relabel them | `labels_clean.parquet` (txn_id, label_clean, keep flag) | a retrain on clean labels does not lower Sep F1 measured on raw labels |
| 9 | S6 | Stage-2 features (research D4, M5) | For each row, aggregate the stage-1 probabilities of its neighbours: the receiver's other inbound rows, the device's other rows, the previous and next hop, the rest of its episode. Mean and max per class. Train rows use **OOF** probabilities only; test rows use test predictions | `features/stage2_v1__half_s0.parquet` | handed to AWS for the stage-2 fit |
| 10 | S7 | Error analysis v2 + weakest slice | Same as task 6 on the stacker OOF. Spend what is left on the worst class | `audit/errors_v2.md` | L2/L3 have a target |
| 11 | S8 | Reproduce run | Launch the clean-clone test on AWS (`../AWS/TASKS.md` task 8) | `reproduce_report.md` | output matches the best upload |

## Handoffs

- **In:** `folds.csv`, `base` (L1); `flow` (L2); `behaviour` (L3); OOF and preds (L2, L3, AWS, Kaggle); `audit/time_consistency.csv` (AWS).
- **Out:** `audit/*` to L1, L2, L3; `labels_clean`, `episodes`, `labelhist` to every trainer and to L1; `stage2` to AWS.

## Don'ts

- Don't let a row's own label, or its own month's labels, into `labelhist`, `episodes` or `stage2` (T4). This is the easiest place in the project to leak.
- Don't relabel on a hunch. Episode-majority and cleanlab only, both in code (T9).
- Don't read the noisy Sep F1 as the truth; report against raw and clean labels (T5).
- Don't train long jobs on the MacBook. Launch them on AWS and keep the laptop free for analysis.
