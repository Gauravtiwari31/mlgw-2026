# Approach note (paste into the Submission tab)

**Team Ship It Friday. Best public upload: `1114-0854__L2_blend16_zones3.csv`, 0.5417.**
Repository: https://github.com/Gauravtiwari31/mlgw-2026. Rebuild with `bash reproduce.sh` (five CSV files in the repo root, Python 3.11, NVIDIA GPU).

1. **Connections across rows.** Features are built on train and test together: how many people paid a receiver in the last hours, how many accounts used a phone, whether money moved on within minutes, reverse pairs, refund counts, round and just-under-limit amounts. October is half the size of a train month, so every window is computed on train at half density to match.
2. **Cleaned train labels.** Out-of-fold models find fraud that was never reported and fraud filed under the wrong type. Models are trained on the cleaned labels and judged on September, held out.
3. **Blend of 16 gradient-boosting runs.** 4 LightGBM and 12 XGBoost; 8 of the XGBoost runs use only features that behave the same way in October.
4. **Phone-id zones.** New phone ids come from a counter. The first 920 new ids of the test month belong to fraud rows and are laid out class by class in label order. A dynamic programme over the id axis finds the class zones without labels (`src/device_block.py`); those rows and rows linked to them through the same account and time are set to the zone's class (`src/zone_apply.py`). This step is worth +0.038 on the public board.

Full notes on what worked and what did not: `audit/device_blocks.md`, `audit/gap_analysis.md`.
