#!/usr/bin/env bash
# Rebuild the best public upload (submissions/1114-0854__L2_blend16_zones3.csv, public 0.5417) from a clean clone.
# Checked 15 Nov 06:33 from a fresh clone: 2 of 260,582 rows differ (GPU XGBoost run-to-run variation). See reproduce_report.md.
# Needs: Python 3.11, `pip install -r requirements.txt`, an NVIDIA GPU for the XGBoost step,
# and the five competition CSV files in the repo root (or data/).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.reproduce
