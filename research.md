# research.md: UPI Fraud Rings (ML Gradient Wars 2026)

Hour-0 research output, following `strategy.md` §3. Written 13 Nov 2026 (Day 1 morning).
Later changes of direction get appended at the bottom with a timestamp.

**Target:** rank 1. The organisers call 0.40 macro-F1 excellent and expect nobody above 0.50, so 100% accuracy is not on the table: training labels are noisy on purpose and some fraud is built to look like legit. A realistic winning range is about 0.45–0.50 private.

---

## Part A. Data audit (measured on the real files, read-only)

| # | Finding | Consequence |
|---|---|---|
| A1 | **Test holds about half of October's activity.** Train: 17.0k rows/day. Test: 8.4k rows/day. Heavy receivers show 0.50x their Sep count in Oct (IQR 0.43–0.57), devices 0.42x, senders 0.44x. | Every count, window and chain feature is half as strong in test as in train. A 4-hop chain is fully visible only about 1 time in 16. **Main hypothesis for where the score is lost.** Fix: density matching (D2). |
| A2 | **`app_version` is a trap.** Train: 4.1–4.8. Test: only 5.0, 5.1, 5.2. In train, fraud rows average 4.06–4.10 vs 4.25 for legit. | A model learns "old version = fraud", which cannot hold in Oct. Drop the raw column. Sep validation cannot detect this. |
| A3 | **`ip_risk_score` was recalibrated.** Sep p5/p50/p95 = 0.04/0.18/0.44. Oct = 0.08/0.21/0.39. It is the strongest raw column (fraud-vs-legit AUC 0.60). | Use the within-month percentile rank, or drop it. Sep validation cannot detect this either. |
| A4 | **Amounts rose 15–17% in Oct** (median ₹464 → ₹541, p95 ₹3,991 → ₹4,736). | Use amount relative to the sender's own history and to the month median. Keep absolute round-number and just-under-limit features. |
| A5 | **147 train rows are dated 1–10 Oct**; 129 are fraud (74 `refund_abuse`). 55–59% of their senders, receivers and devices appear in test. Test has 46 rows dated Nov. | Rows look assigned by episode start month, so episodes spill across the boundary. These labelled rows sit next to test rows: use them through code (neighbour features), never by hand. |
| A6 | **Device farms:** devices with 32–56 accounts. Of 2,241 train rows on devices with ≥10 accounts, 85% are `velocity_bot`. Test has 169 such rows. | Accounts-per-device is close to a rule for `velocity_bot`. |
| A7 | **Single-row signal is weak.** Fraud-vs-legit AUC: ip_risk 0.60, amount 0.58, new_payee 0.56, the rest about 0.50. | The signal is in how rows connect, as the problem statement says. |
| A8 | `qr_vpa_mismatch` = 1 on 30% of `qr_swap` rows vs 0.3% of legit, but null on 64% of rows. `merchant_category` null on 36% (P2P). | Strong but sparse. Keep null as its own value. |
| A9 | 10,667 test receivers received fraud in train. 25,719 test senders sent fraud in train (many are victims). 91.5% of test senders, 79% of receivers, 87% of devices appear in train. | Entity label history is a strong feature, with leakage care (trap T4). |
| A10 | `sim_swap`: median SIM age 3,733 h vs 6,214 h legit, 30 s login-to-pay. `velocity_bot`: 27 s, median amount ₹356. `refund_abuse`: 950 of 3,305 rows are `REFUND` type. `phishing_collect`: 1,821 of 6,697 are `COLLECT`. | Seeds for per-class features. |
| A11 | Class sizes in train: legit 96.31%, mule 0.66%, velocity 0.52%, qr_swap 0.46%, layering 0.44%, phishing 0.43%, sim_swap 0.26%, ATO 0.26%, refund 0.21%, collusion 0.16%, fake_cashback 0.16%, loan_app 0.13%. Stable across Jul/Aug/Sep. | The smallest class has about 2,000 rows: enough to learn, noisy to validate. |

**Open questions for the audit owner (L4, the MacBook):**
- Median `mule_account` rows per receiver is 1, and median `velocity_bot` rows per device is 1. The descriptions suggest bursts. Is that label noise, the mule acting as *sender* in onward transfers, or bots rotating devices? Check before trusting the descriptions.
- Is the test half a random row sample, or are whole episodes kept or dropped together? Compare burst-size distributions Sep vs Oct.

---

## Part B. The four research tracks

```
Track: A (past winners)
Similar comps found: IEEE-CIS Fraud Detection (Kaggle); IBM AML synthetic benchmark (NeurIPS 2023);
  Elliptic Bitcoin; SAML-D (28 AML typologies); temporal-motif studies on Venmo/JPMC.
What won: GBDT (XGB + LGBM + CatBoost) on engineered aggregates. On IBM AML, graph-pattern
  counts as extra columns + XGBoost reach about 0.63 minority F1, level with the best GNNs.
  On Elliptic, Random Forest on features beat GCN.
CV used by winners: IEEE 1st place used GroupKFold by month plus a late-time holdout.
Key features / tricks: IEEE built a client UID and used group mean/std/count over it, never the
  raw UID. Time-consistency screen: single-feature model trained on first month, tested on last,
  drop if AUC <= 0.5. AML: fan-in, fan-out, scatter-gather, cycles, pairs of consecutive
  transactions (dual-edge motifs).
Post-processing: IEEE averaged predictions per UID.
Risks / traps: GNN gains on Elliptic were largely leakage under a strict protocol. Structural
  features can become "high activity" detectors.
Recommendation for us: GBDT on hand-built, time-windowed relational features first. Our "UID"
  is the fraud episode. A GNN is only one more stacker input.
```

```
Track: B (metric tricks, macro-F1)
Similar comps found: multiclass macro-F1 write-ups; label-shift literature.
What won: class-weight scaling of probabilities tuned on OOF, on top of a calibrated model.
CV used by winners: weights fitted on one OOF set and checked on another.
Key features / tricks: predict argmax(w_c * p_c), 12 weights found by coordinate ascent.
  Class weights about inverse-sqrt(frequency) in training. One-vs-rest binaries as stacker inputs.
Post-processing: prior-shift correction: estimate test class priors with EM (Saerens 2002) after
  calibration, or BBSE, then fold into the weights. Episode-level probability averaging.
Risks / traps: public LB is about 78k rows, so 100–250 rows per rare class. Per-class F1 moves
  about +-0.03 and macro-F1 about +-0.01 by chance. Gaps under 0.01 are noise.
Recommendation for us: weighted argmax + prior shift + episode smoothing, all fitted on OOF.
  Test labels are clean and train fraud is under-reported, so expect more fraud in test than
  the 3.7% in train.
```

```
Track: C (domain and industry practice)
Similar comps found: n/a (industry track).
What won: NPCI scores every UPI payment in real time (reported 20 ms SLA). RBI's MuleHunter.AI
  flags mule accounts from transaction + account data, so industry treats mules as an
  account-level problem. Stripe Radar: hundreds of features, mostly network-wide aggregates.
  NVIDIA blueprint: GraphSAGE embeddings into XGBoost.
CV used by winners: n/a.
Key features / tricks: velocity windows (5 min, 30 min, 1 h per device/card); mule red flags:
  throughput ratio in vs out, many unrelated senders then fast outflow, dormant account waking
  up, shared devices; graph practice: connected components over account-device links,
  Louvain to split giant components.
Post-processing: n/a.
Risks / traps: rings rotate devices to break links, so no single link type is enough.
Recommendation for us: industry can only look backwards in time. We label a whole month
  offline, so build every window both before AND after the row ("did the money leave ten
  minutes later?").
```

```
Track: D (data audit and drift)
Similar comps found: LANL Earthquake 1st place (made train look like test); Uber adversarial
  validation paper; IEEE time-consistency screen.
What won: making train features match test conditions, dropping features that drift.
CV used by winners: forward-in-time validation.
Key features / tricks: adversarial validation (train-vs-test classifier), then fix or drop the
  top features. Prefer ratios, means, flags and time gaps over raw counts.
Post-processing: n/a.
Risks / traps: findings A1–A5 above. app_version and ip_risk cannot be judged by Sep validation.
Recommendation for us: density-match train features to test (D2), drop raw app_version,
  rank ip_risk within month, normalise amounts.
```

**Label noise (cuts across tracks).** Two kinds: unreported fraud labelled `legit`, and fraud filed under the wrong type. Plan:
1. Episode-majority relabel: labels follow episodes, so a stray `legit` inside a clear fraud episode is probably unreported fraud.
2. Confident learning (cleanlab) on OOF probabilities. **Drop** flagged rows rather than relabel them.
3. Sep labels are noisy too, so measured F1 understates the truth. Report F1 against raw and cleaned Sep labels.

---

## Part C. The eight merge decisions (strategy.md §3.6)

### D1. Problem type and metric
12-class, single-label classification per transaction. Structurally: edge classification on a temporal multigraph (nodes: sender, receiver, device, merchant). Metric: **macro-F1 over 12 labels**. `legit` gives about 0.98 for free, so the score is decided by the 11 fraud classes. 0.40 overall means an average fraud-class F1 of about 0.35.

### D2. CV scheme (frozen in `folds.csv`)
- **Honest score:** train Jul–Aug, validate Sep. Report macro-F1 and per-class F1.
- **OOF for stacking:** 3 folds by month (Jul, Aug, Sep).
- **Density matching:** each train row gets a random half flag (two seeds: `half_s0`, `half_s1`). Neighbour rule for every relational feature:
  - same month: only rows in the **same half** count
  - earlier months: **all** rows count
  - later months: none
  - test rows: all test rows count, plus the 147 Oct-dated train rows
- **The deciding test (L4):** model trained on full-density features vs half-density features, both scored on half-density Sep.
- **Final fit:** all of Jul–Sep.

### D3. Where the score will be lost
1. **Density mismatch (A1).** Measured by the L4 A/B test.
2. **Drifting columns (A2, A3).** Measured by adversarial validation plus one paired upload.
3. **Fraud types that look alike** (mule vs layering vs ATO vs sim_swap). Measured by the Sep confusion matrix.
4. **Label noise.** Measured by cleanlab flag rates per class.

Day-1 error analysis ranks these by per-class F1, and the effort goes to the worst class.

### D4. Model lineup for Day 1
| # | Model | Machine |
|---|---|---|
| M1 | LightGBM multiclass, class weights (main line) | L2 |
| M2 | XGBoost multiclass, GPU | L2 |
| M3 | CatBoost multiclass, GPU | L3 |
| M4 | 11 one-vs-rest LightGBM binaries | AWS spot |
| M5 | Stage-2 collective LightGBM on neighbour-aggregated stage-1 OOF probabilities | features on L4, fit on AWS (Day 2) |
| M6 | MLP / TabM on the same features | L3 (night) |
| M7 | GNN edge classifier (optional) | L2 (night) |

Final pipeline: stacker over all OOF probabilities → weighted argmax → prior adjustment → episode smoothing.

### D5. Feature ideas, ranked by gain / effort
Windows are built before and after the row at 1 min, 5 min, 30 min, 1 h, 6 h, 24 h, 7 d.

| Rank | Group | Target classes | Gain / effort | Owner |
|---|---|---|---|---|
| 1 | Receiver fan-in (distinct senders, new_payee share) and pass-through (out ÷ in, delay to next outflow) | mule | high / low | L2 |
| 2 | Device and sender bursts, inter-arrival mean/std, accounts per device | velocity_bot | high / low | L3 |
| 3 | Reverse edge: receiver sent this sender ₹1–₹10 shortly before | fake_cashback | high / low | L2 |
| 4 | Refund counts per sender, refund ÷ purchases, refund matches earlier purchase | refund_abuse | high / low | L3 |
| 5 | Pair repeat counts, evening hours, `lending` category | loan_app | high / low | L3 |
| 6 | SIM changed since last payment, new device, device_city ≠ home_city, amount ÷ balance, night | sim_swap, ATO | high / low | L3 |
| 7 | Next-hop and previous-hop amount match (0.85–1.0x within 30 min), chain length, 2-hop skip | layering | high / medium | L2 |
| 8 | Entity label history from other months | all | high / low | L4 |
| 9 | Episode id, size, duration | all | high / medium | L4 |
| 10 | Just-under-round amount, collect fan-in at receiver | phishing_collect | medium / low | L3 |
| 11 | `QR` paid to a P2P receiver, qr_vpa_mismatch, QR fan-in | qr_swap | medium / low | L2 |
| 12 | Merchant round-number share, sender concentration, volume spike vs own baseline | collusion | medium / medium | L2 |
| 13 | Stage-2 neighbour probabilities | all | medium / medium | L4 |
| 14 | GNN embeddings | all | unknown / high | L2 |

Base columns for everyone: hour, weekday, amount features, account/merchant joins, ip_risk as within-month rank. **No raw `app_version`.**

### D6. Metric-specific post-processing to test
1. Weighted argmax, weights by coordinate ascent on OOF.
2. Prior shift by EM on calibrated probabilities.
3. Episode probability averaging.
4. One LB probe for the overall fraud boost.

### D7. Known traps
| # | Trap |
|---|---|
| T1 | Full-density counts in train vs half-density in test |
| T2 | Raw `app_version`, raw `ip_risk_score` |
| T3 | Absolute amount drift |
| T4 | Label-history leakage: a row's own label or own-month labels in its features |
| T5 | Treating noisy Sep F1 as truth |
| T6 | Chasing LB gaps under 0.01 |
| T7 | Risky probe uploads: the best public upload becomes the final |
| T8 | Late reproducibility failure |
| T9 | Hand-labelling test rows (banned). Everything goes through code. |

### D8. Compute assignment
See `work/README.md`. Four machines plus cloud:

| Machine | Work |
|---|---|
| L1, Goro's laptop (no GPU) | Lead, CV harness, base features, uploads, stacker |
| L2, RTX 4060 | Flow features, main GBDT line, overnight GNN |
| L3, RTX 3050 | Behaviour features, CatBoost, neural net |
| L4, MacBook | Audit, drift, density A/B test, episodes, label cleaning, stage-2 features |
| AWS | One-vs-rest binaries, feature screen, stage-2 fit, full rebuilds, final retrains, reproduce test |
| Kaggle (4 accounts) | Seeds and sweeps |

---

## Sources

Several items below are secondary write-ups (news, vendor blogs, a blog summary of a Kaggle solution), flagged as such. Check the primary source before relying on a specific number.

**Industry**
- [eWeek: India's payments network turns to AI](https://www.eweek.com/news/npci-ai-payments-fraud-rules-apac-india/) (news)
- [Business Standard: MuleHunter.ai explained](https://www.business-standard.com/amp/finance/personal-finance/explained-rbi-has-a-new-ai-tool-mulehunter-ai-to-reduce-digital-frauds-124120900250_1.html) (news)
- [Inc42: RBI rolls out AI to detect mule accounts](https://inc42.com/buzz/rbi-rolls-out-ai-to-detect-mule-accounts) (news)
- [ZenML case study: fraud detection at national payment scale](https://www.zenml.io/llmops-database/ai-powered-fraud-detection-at-national-payment-infrastructure-scale) (secondary; source of the 20 ms figure)
- [LexisNexis / Metro Bank mule case study](https://risk.lexisnexis.co.uk/-/media/files/business/case-study/lnrs_metro_bank_cs_uk.pdf)
- [Bureau: money mule detection](https://bureau.id/resources/blog/money-mule-detection) (vendor)
- [Stripe: How we built it: Stripe Radar](https://stripe.com/blog/how-we-built-it-stripe-radar)
- [NVIDIA financial fraud detection blueprint](https://build.nvidia.com/nvidia/financial-fraud-detection)
- [NVIDIA: GNNs for fraud detection](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/)
- [Neo4j: fraud detection with Graph Data Science, part 2](https://neo4j.com/developer-blog/exploring-fraud-detection-neo4j-graph-data-science-part-2/)
- [NebulaGraph: graph algorithms for risk control](https://nebula-graph.io/posts/transforming-risk-control-using-real-world-graph-algorithms-with-nebulagraph)

**Past winners and benchmarks**
- [NVIDIA blog on the IEEE-CIS 1st place solution](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution) (source for UID, month GroupKFold, time-consistency screen, per-UID averaging)
- [Altman et al., Realistic Synthetic Financial Transactions for AML](https://arxiv.org/abs/2306.16424)
- [Egressy et al., Provably Powerful GNNs for Directed Multigraphs](https://arxiv.org/abs/2306.11586)
- [MEGA-GNN](https://arxiv.org/abs/2412.00241)
- [Weber et al., Anti-Money Laundering in Bitcoin (Elliptic)](https://arxiv.org/abs/1908.02591)
- [2026 Elliptic re-evaluation, arXiv 2604.19514](https://huggingface.co/papers/2604.19514) (preprint)
- [Temporal Motifs for Financial Networks](https://ar5iv.labs.arxiv.org/html/2301.07791)
- [D-EMAML: dual-edge motif networks for AML](https://ojs.aaai.org/index.php/AAAI/article/view/38499)
- [SAML-D dataset paper](https://eprints.bournemouth.ac.uk/40982/)
- [Column Networks for Collective Classification](https://arxiv.org/pdf/1609.04508) (stacked learning)

**Metric, shift, noise**
- [Class-weighted coordinate search for F1](https://arxiv.org/pdf/2004.04715)
- [Thresholding classifiers to maximize F1](https://arxiv.org/pdf/1402.1892)
- [Alexandari et al., EM with bias-corrected calibration for label shift](https://arxiv.org/abs/1901.06852)
- [Šipka et al., Hitchhiker's Guide to Prior-Shift Adaptation](https://openaccess.thecvf.com/content/WACV2022/papers/Sipka_The_Hitchhikers_Guide_to_Prior-Shift_Adaptation_WACV_2022_paper.pdf)
- [Northcutt et al., Confident Learning](https://arxiv.org/pdf/1911.00068)
- [cleanlab: mislabeled tabular data and XGBoost](https://cleanlab.ai/blog/label-errors-tabular-datasets)
- [GBDTs on tabular data with label noise](https://arxiv.org/abs/2409.08647)
- [Uber: adversarial validation for concept drift](https://arxiv.org/pdf/2004.03045)
- [LANL Earthquake 1st place write-up](https://medium.com/@ph_singer/1st-place-in-kaggle-lanl-earthquakeprediction-competition-15a1137c2457)
- [RealMLP: Better by Default](https://arxiv.org/abs/2407.04491)
- [TabM](https://arxiv.org/html/2410.24210v1)

---

## Change log
- 13 Nov, Day 1 morning: first version. RTX 4090 not available; compute replanned around RTX 4060 + RTX 3050 + CPU laptops + AWS + Kaggle.
- 13 Nov, Day 1 morning: one MacBook replaces the three CPU laptops (L4–L6). The MacBook keeps audit, episodes, labels and stage-2 features; the unattended training jobs (one-vs-rest binaries, time-consistency screen, stage-2 fit) moved to AWS.
- 13 Nov, 11:50 (L2, `audit/density_ab.md`): **density decision settled. All relational features are built at half density (`half_s0`, with `half_s1` as the second seed).** Half-density training beat full-density training by +0.025 macro-F1 on half-density Sep (all 3 seeds). A naive full-density CV would have overstated the score by 0.045. The `full` view is kept only for diagnostics.
- 13 Nov, 11:40 (L1): single-row baseline `L1_lgbm_base1_s0`: Sep macro-F1 0.385 with plain argmax, 0.398 with class weights fitted on Jul+Aug. Every fraud class is mostly lost to `legit` (recall 20–33%), so recall, not type confusion, is the main gap.
- 13 Nov, 12:30 (L1): **EM prior shift tested, no gain; left off by default.** Calibration works (Sep log loss 0.434 → 0.185, calibrated fraud share 3.65% vs 3.66% in the labels) and EM estimates the fraud share accurately on Sep subsamples with a known shift (2x: true 7.06%, EM 7.23%; 0.5x: true 1.87%, EM 1.64%). But rescaling by the estimated priors did not raise macro-F1 in any test (−0.003, −0.001, −0.014), and even the true priors gave at most +0.002. The class weights already absorb it. EM puts the test month's fraud share at 3.86% vs 3.68% in train: little overall shift. Its per-class ratios (merchant_collusion 2.0x, velocity_bot 0.4x) are probably feature drift, not real prior change, so they are not used. Predicting about 1.5% fraud when the labels hold 3.7% is the F1-optimal trade-off for a weak model, not a bug. The remaining lever is `--fraud-boost` for under-reported train fraud, which only a leaderboard probe can set (research D6, item 4).
- 13 Nov, 12:45 (L1): stacker tested on two models (base, base + six counts). After post-processing: best single 0.4247, mean 0.4236, LightGBM stack 0.4256, **logistic-regression stack 0.4280**, the default. Plain-argmax Sep scores are not comparable between stacks and inputs.
