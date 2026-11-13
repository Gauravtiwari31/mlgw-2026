"""L2 environment check: package versions, GPU training in XGBoost / CatBoost / LightGBM, data load.

Run from the project root:  .venv\\Scripts\\python.exe work/L2_rtx4060_flow_main/check_env.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N, D, K = 400_000, 40, 12


def section(name):
    print(f"\n=== {name} ===", flush=True)


def gpu_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    return out


section("versions")
print("python", sys.version.split()[0], "| cores", os.cpu_count())
for mod in ["polars", "pandas", "pyarrow", "numpy", "sklearn", "lightgbm", "xgboost", "catboost", "optuna"]:
    m = __import__(mod)
    print(f"{mod:10s} {m.__version__}")
try:
    import torch
    print(f"torch      {torch.__version__} | cuda available: {torch.cuda.is_available()}"
          + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
except ImportError:
    print("torch      not installed (only needed for the overnight GNN)")

rng = np.random.default_rng(0)
X = rng.standard_normal((N, D), dtype=np.float32)
y = (np.abs(X[:, :4]).sum(1) * 1.7).astype(int) % K
print(f"synthetic: {X.shape}, {K} classes")
results = {}

section("xgboost")
import xgboost as xgb
for device in ["cuda", "cpu"]:
    try:
        t = time.time()
        m = xgb.XGBClassifier(n_estimators=50, max_depth=8, tree_method="hist", device=device,
                              max_bin=256, verbosity=1)
        m.fit(X, y)
        results[f"xgboost {device}"] = f"OK {time.time() - t:.1f}s"
        if device == "cuda":
            print("gpu mem after fit:", gpu_mem())
    except Exception as e:  # noqa: BLE001
        results[f"xgboost {device}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"xgboost {device}:", results[f"xgboost {device}"], flush=True)

section("catboost")
from catboost import CatBoostClassifier
for task in ["GPU", "CPU"]:
    try:
        t = time.time()
        m = CatBoostClassifier(iterations=50, depth=8, loss_function="MultiClass", task_type=task,
                               verbose=0, allow_writing_files=False)
        m.fit(X, y)
        results[f"catboost {task}"] = f"OK {time.time() - t:.1f}s"
    except Exception as e:  # noqa: BLE001
        results[f"catboost {task}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"catboost {task}:", results[f"catboost {task}"], flush=True)

section("lightgbm")
import lightgbm as lgb
for device in ["gpu", "cpu"]:
    try:
        t = time.time()
        m = lgb.LGBMClassifier(n_estimators=50, num_leaves=127, device_type=device, verbose=-1)
        m.fit(X, y)
        results[f"lightgbm {device}"] = f"OK {time.time() - t:.1f}s"
    except Exception as e:  # noqa: BLE001
        results[f"lightgbm {device}"] = f"FAIL {type(e).__name__}: {str(e)[:200]}"
    print(f"lightgbm {device}:", results[f"lightgbm {device}"], flush=True)

section("data load (polars)")
import polars as pl
import psutil
for name in ["train.csv", "test.csv", "accounts.csv", "merchants.csv", "sample_submission.csv"]:
    p = ROOT / name
    if not p.exists():
        p = ROOT / "data" / name
    if not p.exists():
        print(f"{name}: MISSING")
        continue
    t = time.time()
    df = pl.read_csv(p)
    print(f"{name:22s} {df.shape!s:16s} {time.time() - t:5.1f}s  {df.estimated_size('mb'):7.0f} MB in RAM")
    if name == "train.csv":
        print(df["label"].value_counts(sort=True))
        print("timestamp range:", df["timestamp"].min(), "->", df["timestamp"].max())
    del df
vm = psutil.virtual_memory()
print(f"RAM: {vm.total / 2**30:.1f} GB total, {vm.available / 2**30:.1f} GB free now")

section("summary")
for k, v in results.items():
    print(f"{k:14s} {v}")
