"""XGBoost (GPU) trainer with the same contract as src.train: same features, labels, weights and folds.

    python -m src.train_xgb --config configs/<lightgbm config>.yaml --run-id <new run id>

1. train Jul+Aug, early stopping on Sep (multi-class log loss) -> Sep OOF
2. Jul and Aug OOF with the same number of rounds
3. all of Jul-Sep -> test predictions (rounds x 1.15)
Writes oof/<run_id>.parquet, preds/<run_id>.parquet, runs/<run_id>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import metric
from .common import N_CLASSES, OOF, PREDS, RUNS
from .train import _git_commit, _save_probs, load_matrix, train_mask_and_weights

PARAMS = {
    "objective": "multi:softprob", "num_class": N_CLASSES, "eval_metric": "mlogloss",
    "tree_method": "hist", "device": "cuda", "learning_rate": 0.05, "max_depth": 7,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.5, "lambda": 5.0, "max_bin": 256,
}


def main() -> None:
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    t0 = time.time()
    meta, X, y, names = load_matrix(cfg)
    keep, w = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    ok = meta["keep"].fill_null(True).to_numpy() if "keep" in meta.columns else np.ones(meta.height, dtype=bool)
    is_test = (meta["split"] == "test").to_numpy()
    params = {**PARAMS, "seed": args.seed}

    sep_all = np.where(fold == 2)[0]
    va = np.where((fold == 2) & ok)[0]
    tr = np.where(keep & (fold != 2))[0]
    dtr = xgb.DMatrix(X[tr], y[tr], weight=w[tr], feature_names=names)
    dva = xgb.DMatrix(X[va], y[va], weight=cfg["_val_weight"][va], feature_names=names)
    b = xgb.train(params, dtr, 3000, evals=[(dva, "sep")], early_stopping_rounds=100, verbose_eval=200)
    best = b.best_iteration + 1
    oof = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    oof[sep_all] = b.predict(xgb.DMatrix(X[sep_all], feature_names=names), iteration_range=(0, best))
    rep = metric.report(y[va], oof[va].argmax(axis=1), f"\nSep validation, {best} rounds (plain argmax)")
    for k in (0, 1):
        va_k = np.where(fold == k)[0]
        tr_k = np.where(keep & (fold != k))[0]
        bk = xgb.train(params, xgb.DMatrix(X[tr_k], y[tr_k], weight=w[tr_k], feature_names=names), best)
        oof[va_k] = bk.predict(xgb.DMatrix(X[va_k], feature_names=names))
        sc = va_k[ok[va_k]]
        print(f"fold {k} OOF macro-F1 = {metric.macro_f1(y[sc], oof[sc].argmax(axis=1)):.4f}", flush=True)
    cv_rows = np.where(fold >= 0)[0]
    _save_probs(OOF / f"{args.run_id}.parquet", meta["txn_id"][cv_rows], oof[cv_rows])
    full = np.where(keep)[0]
    bf = xgb.train(params, xgb.DMatrix(X[full], y[full], weight=w[full], feature_names=names), int(best * 1.15) + 1)
    te = np.where(is_test)[0]
    _save_probs(PREDS / f"{args.run_id}.parquet", meta["txn_id"][te], bf.predict(xgb.DMatrix(X[te], feature_names=names)))
    cv_ok = cv_rows[ok[cv_rows]]
    log = {"run_id": args.run_id, "model": "xgboost", "config": args.config, "commit": _git_commit(),
           "features": cfg["features"], "view": cfg.get("view"), "labels": cfg.get("labels"), "seed": args.seed,
           "best_rounds": int(best), "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"],
           "oof_macro_f1": metric.macro_f1(y[cv_ok], oof[cv_ok].argmax(axis=1)), "minutes": round((time.time() - t0) / 60, 1)}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{args.run_id}.json").write_text(json.dumps(log, indent=2))
    print(f"all-months OOF macro-F1 = {log['oof_macro_f1']:.4f}; done in {log['minutes']} min -> runs/{args.run_id}.json")


if __name__ == "__main__":
    main()
