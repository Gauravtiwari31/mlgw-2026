# Reproduce report

Fresh clone of `main` at `cbe14f4` in `E:\mlgw_repro2`, the five data files copied in, `python -m src.reproduce`,
15 Nov 2026 05:17-06:33 IST, RTX 4060 laptop (L2).

Target: `submissions/1114-0854__L2_blend16_zones3.csv` (public 0.5417, the team's best upload).

| Step | Result |
|---|---|
| folds, base_v1, labels_clean --from-changes | ok (15 s) |
| flow / behaviour / behaviour2 in fw_s0 and fw_s1, behaviour3, label history | ok (6.7 min) |
| 4 LightGBM | ok (25 min; the first run shared the CPU with another job, the others took about 3 min each) |
| 12 XGBoost on the GPU (4 plain, 8 with the time-consistent feature screen) | ok (44 min) |
| blend of the 16 runs | ok (6 s) |
| phone-id zones (`src.device_block`) | ok (10 s) |
| zones and linked rows applied (`src.zone_apply`, rules R1 to R4) | ok (3 s) |

**Result: 2 of 260,582 rows differ** from the uploaded file (qr_swap -> refund_abuse 1, velocity_bot -> mule_account 1).
That is GPU XGBoost run-to-run variation; every other step is deterministic.

Run it yourself: `bash reproduce.sh` (Python 3.11, `pip install -r requirements.txt`, an NVIDIA GPU, the five
competition CSV files in the repo root or `data/`). About 75 minutes on this laptop.
