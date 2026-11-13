# Gap analysis: why we show 0.45 and the leaderboard shows 0.65

13 Nov, 13:25, L2. Every number below was measured on our data today. The scripts are being
turned into `src/labels_clean.py` and a corrected feature view; until then treat this file as
the record.

## The short version

1. **We have not uploaded anything.** `submissions/INDEX.csv` and `uploads_log.csv` have no
   public score. Our 0.4476 is September validation against noisy training labels. The leader's
   0.6457 is the public leaderboard against clean October labels. They are different rulers.
2. **63% of the fraud labels in train are ordinary legit rows with a fraud label on them.**
   Against cleaned September labels the same models score about 0.79, not 0.44.
3. **Fraud episodes are whole in the test month.** Only the legit background is halved. The
   half-density rule is wrong for fraud-pattern features. The density A/B test (L2) could not
   see this, because it halved September's fraud rows too.
4. **`ip_risk` carries almost no signal in October**, even as a within-month rank.

## 1. Label noise

A cross-fitted model (4 folds, all columns including raw `ip_risk_score` and `app_version`,
which is fine for cleaning train labels) scores every train row for "does this row carry a
fraud label". Fraud-labelled rows split into two groups with almost nothing between:

| Cross-fitted P(fraud label) | Fraud-labelled rows | Share | Legit-labelled rows | Fraud rate in bin |
|---|---|---|---|---|
| 0.01 to 0.05 | 35,948 | 62.5% | 1,489,879 | 0.023 to 0.024 |
| 0.05 to 0.50 | 1,288 | 2.2% | 13,583 | 0.035 to 0.56 |
| 0.50 to 1.00 | 20,283 | 35.3% | 2,902 | 0.84 to 0.88 |

- The low group has the same fraud rate in every bin (2.3 to 2.4%): a flat chance that any
  ordinary row gets a fraud label. A separate model cannot tell these rows from legit rows
  (AUC 0.505), and every column matches legit: P2P share 0.359 vs 0.361, median amount 458 vs
  460, new payee 0.225 vs 0.227, median IP risk 0.182 vs 0.180, app version 4.244 vs 4.250.
- It is the same in every class: 58% to 71% of each fraud class is in the low group.
- The 2,902 legit-labelled rows in the high group are the unreported fraud.
- Cleaning rule used here: fraud label with P < 0.10 becomes legit (36,304 rows, 63.1%); fraud
  label with P 0.10 to 0.30 (433 rows) and legit label with P >= 0.50 (2,902 rows) are left
  out. Fraud share goes from 3.68% to 1.36%.

Consequence: against raw September labels no model can pass about 0.5, and real improvements
barely move the number. That is why every run sat between 0.43 and 0.45.

## 2. What the same models score against cleaned September labels

base_v1 + flow_v1, LightGBM, lr 0.03, 80 rounds, train Jul+Aug, plain argmax.

| Trained on | ip_risk_rank | Sep vs raw labels | Sep vs clean labels |
|---|---|---|---|
| raw labels | yes | 0.4424 | 0.7835 |
| raw labels | no | 0.4414 | 0.7793 |
| clean labels | yes | 0.4430 | 0.7945 |
| clean labels | no | 0.4426 | 0.7887 |

Caveat: the cleaned labels come from a model that shares features with the models being
scored, so the clean score is somewhat optimistic. The October drift below is not in it either.
Only an upload gives the real number.

Per class against clean labels (clean labels, no ip_risk_rank, half view): qr_swap 0.914,
velocity_bot 0.905, layering 0.879, loan_app 0.846, mule 0.841, collusion 0.798, phishing 0.778,
refund 0.733, sim_swap 0.651, account_takeover 0.590, fake_cashback 0.529. The errors that
remain are between fraud types (sim_swap with account_takeover, phishing with fake_cashback,
mule with layering), not fraud against legit.

## 3. Density: fraud episodes are whole in test

Label-free patterns per 10,000 rows. If test were a row-level half sample, the test month would
match the half view. It matches the full view.

| Pattern (rows that must be visible) | Train, full view | Train, half view | Test month |
|---|---|---|---|
| QR paid to a person (1) | 14.06 | 14.06 | 12.23 |
| Reverse payment in the pair within 24 h (2) | 6.49 | 3.55 | 6.64 |
| Next hop 0.85 to 1.0x within 30 min (2) | 10.36 | 4.95 | 9.24 |
| Next and previous hop both match (3) | 7.12 | 1.70 | 6.21 |
| P2P receiver with 2+ other payments in 24 h (3) | 30.99 | 25.36 | 31.95 |
| P2P receiver with 5+ other payments in 24 h (6) | 22.92 | 17.38 | 27.73 |

Merchant traffic really is halved (141 vs 280 payments per receiver per day), so the legit
background is thinner but each fraud episode is complete.

Train view against September view, clean labels, no ip_risk_rank:

| Train view | Sep half view | Sep full view |
|---|---|---|
| half | 0.7887 | 0.7900 |
| full | 0.7311 | 0.8261 |

- A half-trained model does not collapse on full-density fraud (0.7900), so the existing
  submissions are not badly hurt.
- A model trained at the right density is +0.036 better (0.8261), mostly on fake_cashback
  (0.53 to 0.69), refund_abuse (0.73 to 0.85) and merchant_collusion (0.80 to 0.86).
- Neither view is exactly the test month. The right one is "fraud whole, legit halved". For
  account, pair and device keys the full view is almost that, because those keys have close to
  no legit background. Merchant keys need the mixed view.

## 4. October drift, checked without labels

Mean within-month rank of each column on pattern rows found without labels (0.5 = no signal):

| Column | Jul | Aug | Sep | Oct (test) |
|---|---|---|---|---|
| ip_risk rank, QR paid to a person | 0.761 | 0.770 | 0.763 | 0.540 |
| ip_risk rank, P2P receiver bursts | 0.741 | 0.757 | 0.763 | 0.537 |
| ip_risk rank, hop matches | 0.758 | 0.755 | 0.763 | 0.574 |
| app_version rank, P2P receiver bursts | 0.285 | 0.279 | 0.274 | 0.485 |
| amount / balance rank, P2P receiver bursts | 0.801 | 0.792 | 0.802 | 0.766 |
| new_payee share, P2P receiver bursts | 0.909 | 0.897 | 0.887 | 0.896 |
| night share, P2P receiver bursts | 0.228 | 0.222 | 0.242 | 0.234 |

`ip_risk_rank` is in `base_v1` and is a top-ten feature in every run. Dropping it costs 0.006
on clean September and removes a feature that is close to noise in October. Amount, new payee
and time-of-day features hold.

## 5. What to change, in order

1. Upload the best existing file now (`submissions/1113-1259__stack_v1.csv`). One upload tells
   us where we really are.
2. `labels_clean.parquet` for everyone: train with `labels: clean`, and report September against
   clean labels from now on.
3. Rebuild fraud-pattern features at fraud-whole density (full view for account, pair and device
   keys; a "fraud whole, legit halved" view for merchant keys).
4. Drop `ip_risk_rank`. Run the label-free October check on every new feature group.
5. Then the type confusions: sim_swap against account_takeover, phishing against fake_cashback.
