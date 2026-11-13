#!/usr/bin/env bash
# Rebuild the best public upload (submissions/1113-1912__L2_lgbm4-xgb1_blend5.csv, public 0.5014) from a clean clone.
# Needs: Python 3.11, `pip install -r requirements.txt`, an NVIDIA GPU for the XGBoost step,
# and the five competition CSV files in the repo root (or data/).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.reproduce
