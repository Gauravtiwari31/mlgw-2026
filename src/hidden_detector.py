"""Detector for 'ordinary-looking' fraud (status 'noise' in labels_clean.parquet) against legit rows.

5-fold XGBoost on the GPU over every feature group plus raw columns and ids. Reports how concentrated the
hidden rows are at the very top of the score list (a prediction needs roughly 25%+ precision to help macro-F1)
and writes the out-of-fold scores to cache/hidden_scores.parquet for inspection.

    python -m src.hidden_detector
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import polars as pl

from .common import CACHE, ROOT, load_features, load_raw

GROUPS = [("base_v1", "full"), ("flow_v1", "fw_s0"), ("behaviour_v1", "fw_s0"), ("behaviour_v2", "fw_s0"), ("behaviour_v3", "fw_s0")]


def main() -> None:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    raw = load_raw()
    cl = pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "status")
    j = raw.select("txn_id", "split").join(cl, on="txn_id", how="left", maintain_order="left")
    m = ((j["split"] == "train") & j["status"].is_in(["noise", "legit"])).fill_null(False).to_numpy()
    y = (j["status"] == "noise").fill_null(False).to_numpy()[m].astype(np.int32)
    extra = raw.select(
        x_ip=pl.col("ip_risk_score"), x_appv=pl.col("app_version"),
        x_snum=pl.col("sender_id").str.slice(2).cast(pl.Float32), x_dnum=pl.col("device_id").str.slice(2).cast(pl.Float32),
        x_rnum=pl.col("receiver_id").str.slice(2).cast(pl.Float32), x_tsec=pl.col("ts").dt.epoch("s").cast(pl.Float64),
        x_sec=pl.col("ts").dt.second(), x_min=pl.col("ts").dt.minute(),
        x_txh=pl.col("txn_id").str.slice(2).str.to_integer(base=16).cast(pl.Float64),
        x_cents=(pl.col("amount") * 100).round(0) % 100, x_bal_cents=(pl.col("balance_before") * 100).round(0) % 100,
        x_dev_own=(pl.col("device_id").str.slice(2).cast(pl.Int64) == pl.col("sender_id").str.slice(2).cast(pl.Int64)).cast(pl.Int8),
    )
    blocks, names = [], []
    for g, v in GROUPS:
        f = load_features(g, v).drop("txn_id")
        blocks.append(f.select(pl.all().cast(pl.Float32)).to_numpy()[m]); names += f.columns
    blocks.append(extra.select(pl.all().cast(pl.Float32)).to_numpy()[m]); names += extra.columns
    X = np.hstack(blocks); del blocks
    ids = j["txn_id"].filter(pl.Series(m))
    fold = np.fromiter((hashlib.md5(f"hid:{t}".encode()).digest()[0] % 5 for t in ids.to_list()), dtype=np.int8, count=len(y))
    print(f"rows {len(y)}, hidden {int(y.sum())} ({y.mean():.4f}), features {X.shape[1]}  ({time.time() - t0:.0f}s)", flush=True)
    p = np.zeros(len(y), dtype=np.float32)
    gain = {}
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "learning_rate": 0.05, "max_depth": 7,
              "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.5, "lambda": 5.0, "max_bin": 256, "seed": 0}
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        b = xgb.train(params, xgb.QuantileDMatrix(X[tr], y[tr], feature_names=names), 300)
        p[va] = b.predict(xgb.DMatrix(X[va], feature_names=names))
        for kk, vv in b.get_score(importance_type="gain").items():
            gain[kk] = gain.get(kk, 0.0) + vv
        print(f"fold {k}: AUC {roc_auc_score(y[va], p[va]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"overall AUC {roc_auc_score(y, p):.4f}; base rate {y.mean():.4f}")
    order = np.argsort(-p)
    for frac in (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        k = max(int(len(p) * frac), 1)
        print(f"  top {frac:>7.2%} ({k:>6} rows): hidden rate {y[order[:k]].mean():.4f}   share of all hidden caught {y[order[:k]].sum() / y.sum():.4f}")
    k = int(len(p) * 0.02)
    print(f"  bottom 2%: hidden rate {y[order[-k:]].mean():.4f}")
    print("top features by gain:", ", ".join(n for n, _ in sorted(gain.items(), key=lambda kv: -kv[1])[:25]))
    CACHE.mkdir(exist_ok=True)
    pl.DataFrame({"txn_id": ids, "hidden": y, "score": p}).write_parquet(CACHE / "hidden_scores.parquet")


if __name__ == "__main__":
    main()
