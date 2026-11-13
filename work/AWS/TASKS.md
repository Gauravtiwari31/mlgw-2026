# AWS: unattended training jobs ($250+ credits)

**Owner:** P4 (operated from the MacBook) · **Role:** every job that is only "fit a model and wait", plus full feature rebuilds, final retrains and the reproduce test.

With one MacBook in place of three CPU laptops, AWS is where the CPU hours come from.

## Setup (once, in S1)

1. Pick one region and stay in it. Create one S3 bucket with the folders `data/ features/ oof/ preds/ audit/ submissions/`.
2. One launch template: Ubuntu, Python 3.11, an instance role with access to the bucket. Its start-up script clones the repo and runs `pip install -r requirements.txt`.
3. Every job follows the same pattern: pull the repo, sync inputs from S3, run **one** command with a config file, sync outputs back to S3, shut down.

```
python -m src.train --config configs/<name>.yaml
```

## Budget

Spend cap: about $200. Keep at least $50 in reserve. Prices below are rough; read the real ones in the console.

| Use | Instance class | Hours | Rough cost |
|---|---|---|---|
| Feature rebuilds, stage-2 fit, final retrains | r7i or r6i 4xlarge (16 vCPU, 128 GB) | ~30 | $35–60 |
| One-vs-rest binaries, feature screen, sweeps (spot) | 3 × c7i 4xlarge | ~15 each | $15–35 |
| GPU overflow, only if L2 is saturated | g4dn or g5 xlarge | ~15 | $10–25 |
| Clean-clone reproduce test | small CPU box | ~3 | under $5 |

## Tasks

| # | Slot | What | How | Output | Done when |
|---|---|---|---|---|---|
| 1 | S1 | Billing alarms | Budget alerts at $50, $120 and $200, emailed to P4 and Goro | alarms live | test email received |
| 2 | S1 | GPU quota request | Request on-demand and spot G/VT vCPU quota now. New accounts often start at 0 and approval takes hours | request filed | noted in the team chat |
| 3 | S3 → S4 | M4: 11 one-vs-rest LightGBM binaries | One spot c7i box, one config per class, each class against all other rows, on base + flow + behaviour. Month-fold OOF | 11 files in `oof/` and `preds/` | L1 can stack them |
| 4 | S4 | Time-consistency screen | Per feature: a one-feature model trained on Jul, scored on Sep, for fraud-vs-legit. Flag features at or below chance (IEEE lesson) | `audit/time_consistency.csv` | L4 folds it into `drop_list.txt` |
| 5 | S4 → S5 | Full feature rebuild | Once `flow_v2` and `behaviour_v2` are merged: rebuild every group for train + test in `half_s0` and `half_s1` on the r-class box | all `features/*` on S3 | checksums in the team chat |
| 6 | S6 | M5: stage-2 LightGBM | Train on base + flow + behaviour + L4's `stage2` features, month-fold OOF | `oof/` + `preds/` | L1 can stack it |
| 7 | S8 | Final multi-seed retrains | Frozen configs from L2 and L3, full Jul–Sep fit, 3–5 seeds each, in parallel on spot | `preds/final_*` | L1 has them for the final stack |
| 8 | S8, **before 02:00 on 15 Nov** | Clean-clone reproduce test | Fresh box, fresh clone, run `reproduce.sh`, compare the output CSV with the best upload row by row | `reproduce_report.md` | zero differing rows, or a list of what differs |

## Handoffs

- **In:** code from GitHub; `features/*` and `folds.csv` from S3; configs from L2, L3, L4.
- **Out:** `oof/*` and `preds/*` to L1 and L4; `audit/time_consistency.csv` to L4; `reproduce_report.md` to L1.

## Don'ts

- Don't leave an instance running idle. Every job script ends with a shutdown.
- Don't run anything without the billing alarms in place.
- Don't put the repo or the bucket in public mode (sharing code between teams is disqualifying).
- Don't store AWS keys in the repo. Use the instance role.
- Don't rely on a spot box for the reproduce test; use on-demand for that one.
