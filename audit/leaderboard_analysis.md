# Leaderboard analysis (13 Nov, evening, L2)

Supersedes the "clean labels" reading in `audit/gap_analysis.md` section 2. The cleaning itself is
still right for training; it is wrong as a picture of the test labels.

## What the uploads say

| Upload | Public | Meaning |
|---|---|---|
| `1113-1401__L2_lgbm_fw-clean_b1-f1-beh12_s0__clean.csv` | 0.4987 (rank 11) | full model |
| `1113-1729__PROBE_only_qr_swap.csv` | 0.1261 | qr_swap F1 on test about 0.53 |

A one-class probe scores (F1_legit + F1_class) / 12, so 0.1261 puts qr_swap F1 at about 0.53.
We predict about 148 qr_swap rows in the public part with high precision, so the public part holds
about 400 qr_swap rows: 1.7 ordinary-looking rows per visible one. Train has 4,508 / 2,680 = 1.68.

**October carries the same ordinary-looking fraud as train (about 2.3% of all rows).**

## Ordinary-looking fraud cannot be found in this data

Checked against legit rows, all equal within noise:
- every row column and its format (amount, balance, SIM, device, login, PIN, IP, app, city, bank, KYC, OS, hour)
- account, phone and receiver id ranges; phone id against its owner's id
- links to visible fraud (same sender, phone, receiver, pair; within 7 days or any month; per class and per role)
- links to other ordinary-looking fraud (same entity, same month or across months)
- time: gaps between such rows match random rows; no daily or hourly clustering
- running state: balance, SIM age, device age and account age are drawn fresh per row
- transaction-id hex digits, duplicates, CSV row order (no relation to labels or time)
- a model trained only to find them on 416 features plus raw columns: AUC 0.530; its top 1% holds 8.7% of them
  (a prediction needs about 27% precision to help F1)

## Consequences

| Ordinary-looking fraud rate | Our model (simulated) | Perfect detector of visible fraud |
|---|---|---|
| 0 (clean) | 0.886 | 1.000 |
| 2.32% (train; probe confirms) | 0.530 | 0.590 |

- With October's rate, the best possible average F1 over the fraud classes is about 0.52. The leader's
  0.6457 needs about 0.61. We cannot explain that score from this data.
- We scored 0.4987 against a simulated 0.530. Part of the gap is that October has fewer velocity bots
  (7 bot phones against about 25 a month) and less collusion: fewer visible rows means a lower F1 under
  the same noise, and no model change fixes that.
- October amount drift (legit +16%, fraud unchanged) costs nothing: replaying it on Sep changes the
  score by +0.004.
- Decision weights tuned under simulated noise: +0.0014. Fraud boost: no gain.
- Velocity bots: September already had both tactics (farms of ~44 accounts, and 3 accounts paying
  20 to 60 times). The model catches both in October.

## What is still worth doing

Under noise each class F1 is about 0.4 to 0.6. The weakest are sim_swap 0.40 and account_takeover 0.40
(type confusion; about 30% of their train labels carry the other type's signature), fake_cashback 0.42,
mule 0.46. Moving sim_swap and ATO from 0.73 to 0.90 on clean labels is worth about +0.015 public.
