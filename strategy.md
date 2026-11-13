# ML Gradient Wars 2026 — Strategy

**Team:** Goro (Gaurav Tiwari) + team · **Event:** ML Gradient Wars 2026 by Hack4Hack · **Goal:** Rank 1. Not top 10, not "good". Rank 1.

> Live round: **13 Nov 10:00 AM IST → 15 Nov 10:00 AM IST** (48 hours). Results: 15 Nov, 5:00 PM IST.

---

## 0. Ground rules (non-negotiable)

1. **Play for rank 1.** A rank like 344 (Amazon MLC 2026) is a loss, not a result.
2. **Research before code.** The first hour goes to studying the problem type and past winning solutions. No architecture gets picked before that.
3. **Find where the score is lost first.** In Amazon MLC it was recall (true matches never entered the candidate list). Here, run error analysis on Day 1 to find the equivalent hole before tuning models.
4. **Compute is never an excuse.** More machines beat one big machine. Kaggle × 4 accounts + Colab + AWS credits + the 4090, all running in parallel.
5. **Final call by a stacker**, not a single model or a hand-tuned cascade.
6. **Validate only on the real leaderboard metric**, with a CV split that copies how test was split.
7. **Every upload is CV-backed.** The best public upload becomes the final, so no public-LB chasing.
8. **Ghonchu's limits, stated upfront:** no GPU in the sandbox, no login to the Hack4Hack workspace. Ghonchu writes and tests code on small samples; heavy training runs on Kaggle, Colab, AWS or the 4090. Upload the dataset/PS to the chat as soon as it drops.

---

## 1. Event facts that shape the strategy

| Item | Detail | What it means for us |
|---|---|---|
| Format | One round, one dataset, one leaderboard, 48 hrs | No second chances. Speed of iteration decides it. |
| Team | 1–4 people | 4 people = 4 parallel workers + sleep rotation |
| Metric | Announced with the PS | Starter kit has a pluggable metric; we lock it in the first 15 min |
| Uploads | Max 10/day → ~20 total | Each upload is planned and logged (see §7) |
| Public LB | Refreshes every 15 min, on a public part of test | Signal is noisy; trust CV over public |
| Final ranking | Private part of test, using each team's **best public upload** | No choosing a final. A lucky overfit upload can become our final and drop. |
| Code | GitHub repo / notebook link submitted from Submission tab before close | Repo must be clean and final before 15 Nov 10:00 AM |
| Verification | Top teams' code must reproduce their best upload | Fixed seeds, saved configs, one `reproduce.sh`. Teams lose here. |
| Prizes | 1st Rs 2,00,000 · 2nd Rs 1,50,000 · 3rd Rs 1,00,000 · 4th Rs 30,000 · 5th Rs 20,000 | Top 3 also get The Data Foundation 3-month internship referral |
| Resources on 13 Nov | PS, train data, test (no labels), sample submission, data dictionary | Read the data dictionary line by line in hour 0 |
| Organiser's rule | Models that generalise win; LB overfitting doesn't | Confirms: CV discipline > public climbing |

---

## 2. Lessons carried over from Amazon ML Challenge 2026

**Our result:** 0.987133, rank 344 of ~29,000. **Rank 1 (HyperNetworks, DTU)** ran almost the same pipeline. We lost on recall and experiment count, not on hardware or model choice.

| Step | Ship It Friday | Rank 1 | Lesson for MLGW |
|---|---|---|---|
| Candidate finding | 2 channels (TF-IDF + e5) | 9 channels | Measure the ceiling of every early stage; widen until misses stop dropping |
| First cut | LightGBM only | LightGBM + small transformer | Mix model families early |
| Unsure cases | 3 cross-encoders on ~4% pairs | Cross-encoder on unsure pairs | Same idea, fine |
| Final decision | Cascade + per-country thresholds | Stacker over all scores | **Stacker, always** |
| Hardware | One RTX 4090 | AWS, many parallel runs | More workers > faster worker |
| Experiments | Few, slow | Many, side by side | Iteration count wins timed contests |

**Where we lost, in order:** (1) recall, (2) too few experiments, (3) no stacker, (4) the weakest slice (France) found but never fixed.

**MLGW translation:** on Day 1 find our "recall" — the biggest source of error (a segment, a class, a feature gap, a leak in CV). On Day 2 fix the weakest slice instead of polishing what already works.

---

## 3. THE RESEARCH PHASE (Hour 0 → Hour 1)

This is the part we skipped at Amazon MLC. It is now a fixed, timed protocol.

### 3.1 What we already know (pre-event research)

- **No past editions of MLGW or Hack4Hack solutions exist online.** It's a new event; there's no organiser-specific history to mine. All research happens on 13 Nov against the problem *type*.
- **Field-wide findings (ML Contests' reviews of ~390–400 competitions per year, 2024 and 2025):**
  - GBDTs (XGBoost, LightGBM, then CatBoost) are still the default winners for tabular and time-series problems, often ensembled with a neural net.
  - Almost all winners use Python; PyTorch dominates deep learning.
  - For text tasks, winning solutions are transformer-based, and in 2025 almost every winning text solution used Qwen in some way.
- **Where winning solutions live (bookmark these before 10 AM):**
  - Kaggle Solutions (Farid Rashidi) — https://github.com/faridrashidi/kaggle-solutions — the largest list of top-performer write-ups, searchable by competition.
  - Kaggle Past Solutions (Eliot Andres) — https://ndres.me/kaggle-past-solutions/ — searchable, sortable.
  - ML Contests "State of ML Competitions" 2025 report — https://mlcontests.com/state-of-machine-learning-competitions-2025 — what won, by problem type.
  - Kaggle competition **Discussion** tabs filtered by "1st place solution", "2nd place", "gold".
  - Papers with Code / arXiv for the specific task name (if it's a known benchmark task).

### 3.2 Hour-0 timeline (10:00 → 11:00 AM IST, 13 Nov)

| Time | Who | Task | Output |
|---|---|---|---|
| 10:00–10:10 | Everyone | Read PS, rules, metric, data dictionary, sample submission. Screenshot everything and paste PS + dictionary into the chat with Ghonchu. | Shared understanding |
| 10:00–10:10 | Data person | Download data to Kaggle dataset (private) + S3 + Drive. Print shapes, dtypes, target distribution, ID columns, time columns. | `data_summary.txt` |
| 10:10–10:15 | Goro | **Classify the problem** (see §3.3). Write the metric's exact formula. | Problem type + metric locked |
| 10:15–10:45 | Split 4 ways | **Parallel research sprint** (§3.4). Each person owns one research track. | 4 short notes |
| 10:15–10:45 | Ghonchu | Run targeted searches for the problem type + metric; draft the CV scheme and baseline config from the data summary. | Research notes + baseline code |
| 10:45–11:00 | Everyone | **Research merge meeting** (§3.6). Pick the plan. | `research.md` committed to repo |

No training starts before 10:45, except the dumb baseline (§4) which can run in the background as a pipeline check.

### 3.3 Step 1: classify the problem (5 minutes)

Answer these in writing:

1. **Task:** binary / multiclass / multilabel / regression / ranking / matching / segmentation / detection / forecasting / generation?
2. **Modality:** tabular / text / image / audio / time series / graph / mixed?
3. **Metric:** exact formula. Is it threshold-dependent (F1, accuracy) or rank-based (AUC, MAP@K)? Does it punish big errors (RMSE) or relative errors (RMSLE, MAPE)?
4. **Size:** rows, columns, text length, image count. Does it fit in RAM? On a T4?
5. **Structure:** is there time? groups (user IDs, store IDs, patients)? duplicates? hierarchy?
6. **Train vs test:** same period? same groups? any IDs overlapping? (feeds adversarial validation)

### 3.4 Step 2: parallel research sprint (30 minutes, 4 tracks)

| Track | Owner | What to find | Where |
|---|---|---|---|
| **A. Past winners** | Person 1 | Top-3 write-ups of 3–5 past competitions with the same task + modality + metric. Note: model families, CV scheme, key features, post-processing, ensemble method. | Kaggle Solutions lists, Kaggle discussion "1st place solution" |
| **B. Metric tricks** | Person 2 | How winners optimised this exact metric (threshold search, target transform, rank averaging, calibration, class weights). | Kaggle discussions + §5 cheat-sheet |
| **C. Domain + features** | Person 3 | What the columns actually mean in the real world; obvious engineered features; known leakage patterns in this kind of data. | Data dictionary, domain blogs, papers |
| **D. Data audit** | Person 4 | Missing values, duplicates, train/test drift, target leakage, ID patterns, adversarial validation AUC. | The data itself (runs code) |

**Search queries to fire (swap in the real task/metric):**
- `kaggle "<task type>" "<metric>" 1st place solution`
- `kaggle <domain e.g. churn / credit default / sales forecast> winning solution`
- `<metric> optimization kaggle trick`
- `site:kaggle.com/competitions "<similar competition name>" discussion`
- `"<task>" state of the art 2025 2026 benchmark`
- `<task> LightGBM vs neural network kaggle`

**Rule for track A:** read the **summary + model list + CV section** of each write-up, not the whole thing. 6 minutes per write-up maximum.

### 3.5 Step 3: what each research note must contain

Each person writes 5–8 bullets, no more:

```
Track: A / B / C / D
Similar comps found: <names + links>
What won: <model families, ensemble type>
CV used by winners: <KFold / GroupKFold / time split / stratified>
Key features / tricks: <list>
Post-processing: <thresholds, clipping, rounding, rank avg>
Risks / traps: <leakage, drift, small data noise>
Recommendation for us: <one line>
```

### 3.6 Step 4: research merge → `research.md` (15 minutes)

Decide and write down, in this order:

1. **Problem type + metric** (exact).
2. **CV scheme** — chosen to copy how test was split (random → StratifiedKFold; groups → GroupKFold; time → forward/time-based split). Confirmed by adversarial validation.
3. **Where we think the score will be lost** (our "recall" hypothesis) and how we'll measure it.
4. **Model lineup** for Day 1 (3–6 model families, one owner each).
5. **Feature ideas ranked** by expected gain / effort.
6. **Metric-specific post-processing** to test.
7. **Known traps** to avoid.
8. **Compute assignment** (which experiment runs on which machine).

`research.md` gets committed. Any later change of direction gets added to it with a timestamp.

### 3.7 Research doesn't stop at hour 1

- **Hour ~12 checkpoint:** 15-minute mini-research on the weakest slice found in error analysis ("how did winners handle rare classes / cold-start / long texts / etc.").
- **Day 2 morning:** 15-minute research on ensembling for this metric (rank average vs logistic stacker vs hill climbing).
- Whenever stuck for more than 1 hour on a problem → stop and search before trying more blind ideas.

---

## 4. Hour 1 → Hour 3: foundation

1. **Adversarial validation** — train a classifier to tell train from test. AUC ≈ 0.5 → random split is fine. AUC high → find the drifting features, drop or fix them, and build CV to mimic the shift.
2. **CV split frozen** — same fold indices saved to disk (`folds.csv`) and used by every model. Never change folds after this point (OOFs must line up for the stacker).
3. **Dumb baseline** — LightGBM default params on raw features. Check CV vs public LB once.
4. **First upload** — baseline. This calibrates CV ↔ public correlation.
5. **Error analysis v1** — where is the baseline wrong? By segment, by class, by feature range, by data source. This is the "recall" question: **what's the ceiling and where's the gap?**

---

## 5. Problem-type playbooks (pick the one that matches the PS)

### Tabular classification / regression (most likely)
- LightGBM + XGBoost + CatBoost on GPU, each with OOF saved.
- Feature engineering > tuning: group aggregates (mean/std/count by category), ratios, differences, frequency encoding, target encoding inside folds only, date parts, interaction features.
- A neural net (MLP / TabM / FT-Transformer style) for ensemble diversity.
- Optuna tuning only after features stop giving gains.
- Seed averaging (3–5 seeds) on final models.

### Text / NLP
- Fine-tune DeBERTa-v3 and a Qwen model (LoRA if large); TF-IDF + linear model as a fast baseline and as a stacker input.
- Max length chosen from token-length distribution; truncate head+tail for long texts.
- Embeddings (e5 / bge / Qwen embeddings) → GBDT as another family.

### Image / vision
- timm backbones (ConvNeXt, EfficientNet, ViT/EVA) with pretrained weights; strong augmentations; TTA at inference.
- Embeddings from a pretrained model → GBDT/kNN as a cheap extra family.

### Time series / forecasting
- Time-based CV only (forward chaining). Lags, rolling stats, calendar features, target transforms (log).
- GBDT on lag features usually beats pure deep models in short competitions.

### Matching / ranking / retrieval (Amazon MLC-type)
- Multi-channel candidate generation (lexical, char n-gram, phonetic, normalised/abbrev-cleaned, numeric/field-specific, dense embeddings). **Measure recall per channel; keep adding until it stops rising.**
- GBDT ranker (LambdaRank) + cross-encoder on unsure pairs + stacker.

### Metric cheat-sheet
| Metric | Trick |
|---|---|
| AUC | Rank-average ensembles; thresholds don't matter |
| F1 / accuracy | Optimise the decision threshold on OOF; per-class thresholds for multiclass F1 |
| Log loss | Calibrate; clip predictions (e.g. 1e-15 to 1-1e-15); temperature scaling |
| RMSE | Watch outliers; consider Huber loss |
| RMSLE / MAPE | Train on log1p(target), invert at the end |
| QWK | Regress, then optimise class cut-points on OOF |
| MAP@K / NDCG | LambdaRank objective; candidate recall is the ceiling |
| Macro metrics | Class weights / resampling; per-class thresholds |

---

## 6. Day 1 and Day 2 run-sheet

### Day 1 (13 Nov, 11 AM → 14 Nov, 10 AM)
- **11:00–15:00:** All 4 machines run different model families on the frozen folds. Ghonchu writes each config; humans launch and monitor.
- **15:00:** Error analysis v2 → identify the biggest error source. Feature work targets that.
- **Evening:** Feature round 2, model retrains. Second mini-research on the weak slice.
- **Night:** Long runs (transformer fine-tunes, big CV, Optuna) on cloud while people sleep in rotation (2 awake, 2 asleep, 4-hour shifts).
- **Uploads Day 1:** up to 10, each one CV-backed (see §7).

### Day 2 (14 Nov, 10 AM → 15 Nov, 10 AM)
- **Morning:** Stacker v1 over all OOFs (logistic/ridge stacker, or LightGBM with shallow depth; compare against rank-average and hill-climbing ensemble).
- **Midday → evening:** Spend leftover effort on the **weakest validated slice**. Seed-average best models.
- **Night:** Final retrains on full data with fixed seeds; reproduce-check runs.
- **15 Nov 6:00–9:00 AM:** Code freeze. Run `reproduce.sh` from a clean clone and confirm the output matches the best upload exactly.
- **By 9:30 AM:** Submit GitHub repo/notebook link from the Submission tab. **Don't leave this for 9:59.**

---

## 7. Upload discipline (~20 uploads total)

**The rule:** upload only when local CV improves meaningfully, or to answer a specific question about CV ↔ LB correlation.

Upload log (shared sheet, filled on every upload):

| # | Time | Git commit | Config | CV score | Public score | Note |
|---|---|---|---|---|---|---|

**Suggested budget:**
- Day 1 (10): 1 baseline, 2–3 to check CV↔LB correlation, rest on real CV improvements.
- Day 2 (10): stacker versions, best seed-averaged models, 2 held back for emergencies.

**Red flag:** public goes up while CV goes down → overfit to the public split. Don't keep it as best. Since best public = final, a bad upload that scores high on public is dangerous. Prefer fewer, solid uploads.

---

## 8. Compute plan

| Source | Capacity | Use |
|---|---|---|
| RTX 4090 (teammate) | Fastest single GPU | Main line, transformer fine-tunes |
| Kaggle × 4 accounts | ~30 GPU hrs/week each (2×T4 or P100), phone-verified | Parallel GBDT/NN experiments, different configs per account |
| Colab (all accounts) | Backup GPU | Overflow runs |
| AWS (credits) | Multiple EC2 GPU instances | Parallel sweeps; data in S3, code from GitHub, `python train.py --config X` per instance |
| Goro's laptop | No discrete GPU | Coordination, logging, submissions, EDA on samples |
| Ghonchu (sandbox) | CPU, no GPU | Writing/testing code on samples, analysis of OOFs and logs |

**Parallel workflow:** data on S3/Kaggle dataset once → code on GitHub → each machine pulls and runs a different config → each writes OOF + test preds + CV score to shared storage → stacker reads all.

**Safety:** AWS billing alert set; shut down idle instances; GPU quota requested before the event.

---

## 9. Tech stack (decided 11 Nov)

Leaderboard scoring, not latency, so **Python only**. Go/Rust/Fastify add hours and zero score.

| Layer | Pick |
|---|---|
| Language | Python 3.11 |
| Data | Polars (fast), pandas fallback |
| Tabular | LightGBM + CatBoost + XGBoost (GPU), stacked |
| DL / NLP | PyTorch, HF transformers, sentence-transformers, timm |
| Retrieval (if matching) | FAISS, rapidfuzz, bm25s |
| Tuning | Optuna |
| Experiment log | Shared sheet/CSV (run id, config, CV, LB); W&B if the team knows it |
| Code sync | One GitHub repo, notebook → script pipeline |
| Serving/demo | Only if the rules ask: FastAPI + ONNX Runtime / Gradio |

---

## 10. Repo layout and reproducibility

```
mlgw-2026/
├── research.md          # hour-0 research decisions (timestamped updates)
├── configs/             # one YAML per experiment
├── src/
│   ├── data.py          # load + clean
│   ├── folds.py         # builds and saves folds.csv (frozen)
│   ├── features.py
│   ├── train.py         # --config path → OOF + test preds + CV score
│   ├── stack.py         # reads all OOFs → final preds
│   └── metric.py        # the exact competition metric
├── oof/                 # saved OOF predictions per run
├── preds/               # saved test predictions per run
├── uploads_log.csv
├── requirements.txt     # pinned versions
└── reproduce.sh         # clean clone → exact best submission
```

- Fixed seeds everywhere (numpy, torch, lightgbm, python `random`).
- Every upload maps to a commit + config.
- `reproduce.sh` is tested from a clean clone on Day 2 night, not on the last morning.

---

## 11. Team roles

| Role | Owner | Responsibilities |
|---|---|---|
| Lead / coordinator | Goro | Research merge, decisions, upload log, submissions, final repo submission |
| Main-line modeller | 4090 owner | Strongest model family + fine-tunes |
| Feature engineer | Teammate 3 | Error analysis → features for the weakest slice |
| Ensemble / infra | Teammate 4 | Kaggle/AWS parallel runs, OOF collection, stacker, reproduce script |
| Code + analysis | Ghonchu | Writes all pipeline code, research searches, OOF/error analysis, stacker code |

(Fill names in once the team is final.)

---

## 12. Final checklist (15 Nov morning)

- [ ] Best upload identified and matched to commit + config
- [ ] `reproduce.sh` reproduces it exactly from a clean clone
- [ ] `requirements.txt` pinned; seeds fixed
- [ ] README explains how to run
- [ ] Repo public/accessible as the rules require
- [ ] Link submitted from Submission tab well before 10:00 AM IST
- [ ] Screenshots of final leaderboard and submission confirmation saved

---

## Sources

- ML Contests, *The State of Machine Learning Competitions* (2025 edition): https://mlcontests.com/state-of-machine-learning-competitions-2025
- ML Contests summary of 400+ competitions in 2024 (GBDTs still win tabular/time-series, often ensembled with DL)
- Kaggle Solutions collection: https://github.com/faridrashidi/kaggle-solutions
- Kaggle Past Solutions: https://ndres.me/kaggle-past-solutions/
- Competitive ML review of ~400 competitions (Qwen in winning text solutions): https://reddit.sentinel-team.org/posts/1r8y1ha/snapshots/2026-02-19T21%3A44%3A19.909861Z
- MLGW 2026 hackathon page and team workspace (Hack4Hack)
- Amazon ML Challenge 2026 rank-1 write-up (Somin Aggarwal, HyperNetworks, LinkedIn)
