#!/usr/bin/env bash
# Rebuild the best public upload (submissions/1113-2140__L2_blend8_beh3.csv, public 0.5019) from a clean clone.
# Checked 13 Nov 22:32 on the previous best (0.5014): 3 of 260,582 rows differ (GPU XGBoost run-to-run variation).
# Needs: Python 3.11, `pip install -r requirements.txt`, an NVIDIA GPU for the XGBoost step,
# and the five competition CSV files in the repo root (or data/).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.reproduce
