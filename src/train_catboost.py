"""Config-driven CatBoost trainer (L3, work/L3 TASKS.md #7, model M3).

    python -m src.train_catboost --config configs/L3_catboost_base1-flow1-beh1_s0.yaml            # full run
    python -m src.train_catboost --config configs/L3_catboost_base1-flow1-beh1_s0.yaml --mode sep # honest score only

Same contracts as src.train (LightGBM): the same load_matrix (features, view, labels: clean,
drop_features), the same row weights and legit downsampling, the same month-fold CV, and the same
outputs (oof/, preds/, runs/<run_id>.json with the same keys plus model=catboost).

CatBoost-specific settings go under `params:` in the config. Defaults here: task_type GPU on
device 0, MultiClass loss, depth 8, border_count 128. Early stopping uses the validation
MultiClass logloss (the L2 early-stopping audit says the stopping rule does not matter);
the honest Sep macro-F1 is reported separately, as always.

VRAM fallback (TASKS.md): if the 4 GB card runs out, lower `border_count`, then set
`task_type: CPU` in the config, then move the run to a Kaggle T4.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from . import metric
from .common import N_CLASSES, OOF, PREDS, RUNS
from .train import _git_commit, _save_probs, load_matrix, train_mask_and_weights

DEFAULT_PARAMS = {
    "loss_function": "MultiClass",
    "task_type": "GPU",
    "devices": "0",
    "learning_rate": 0.05,
    "depth": 8,
    "border_count": 128,  # GPU default; the first thing to lower if VRAM runs out
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bayesian",
    "verbose": 200,
    "allow_writing_files": False,
}


def fit(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
        cfg: dict, names: list[str], rounds: int | None = None):
    """Train one CatBoost model. With rounds=None: early stopping on the Sep fold."""
    from catboost import CatBoostClassifier, Pool

    params = {**DEFAULT_PARAMS, **cfg.get("params", {}), "random_seed": cfg.get("seed", 0)}
    cat = [n for n in names if n.startswith("cat_")]
    tr_pool = Pool(X.iloc[tr_idx], y[tr_idx], weight=w[tr_idx], feature_names=names, cat_features=cat)
    if rounds is None:
        va_pool = Pool(X.iloc[va_idx], y[va_idx], feature_names=names, cat_features=cat)
        model = CatBoostClassifier(iterations=cfg.get("num_boost_round", 3000), **params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True,
                  early_stopping_rounds=cfg.get("early_stopping", 100))
        return model, int(model.get_best_iteration()) + 1
    model = CatBoostClassifier(iterations=rounds, **params)
    model.fit(tr_pool)
    return model, rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["all", "sep"], default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = cfg["run_id"]
    t0 = time.time()

    meta, X, y, names = load_matrix(cfg)
    keep, w = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    is_test = (meta["split"] == "test").to_numpy()

    # CatBoost wants the cat_* code columns as integers, not the float32 the shared loader makes.
    X = pd.DataFrame(X, columns=names)
    for n in names:
        if n.startswith("cat_"):
            X[n] = X[n].astype("int32")
    print(f"run {run_id}: {len(names)} features, {int(keep.sum())} training rows after sampling, "
          f"task_type={cfg.get('params', {}).get('task_type', 'GPU')}")

    # 1. honest score: train Jul+Aug (+ Oct-dated train rows), validate Sep
    va = np.where(fold == 2)[0]
    tr = np.where(keep & (fold != 2))[0]
    try:
        booster, best = fit(X, y, w, tr, va, cfg, names)
    except Exception as e:
        raise RuntimeError(
            f"fit failed ({e}). VRAM fallback (TASKS.md): lower params.border_count, "
            f"then set params.task_type: CPU, then move the run to a Kaggle T4."
        ) from e
    oof = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    oof[va] = booster.predict_proba(X.iloc[va])
    rep = metric.report(y[va], oof[va].argmax(axis=1), f"\nSep validation, {best} rounds (plain argmax)")
    imp = sorted(zip(booster.get_feature_importance(), names), reverse=True)[:25]
    print("top features by importance:", ", ".join(n for _, n in imp))

    log = {
        "run_id": run_id, "config": args.config, "commit": _git_commit(), "model": "catboost",
        "features": cfg["features"], "view": cfg.get("view", "half_s0"),
        "seed": cfg.get("seed", 0), "best_rounds": int(best),
        "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"], "n_features": len(names),
    }
    if args.mode == "all":
        # 2. OOF for Jul and Aug, same number of rounds
        for k in (0, 1):
            va_k = np.where(fold == k)[0]
            tr_k = np.where(keep & (fold != k))[0]
            b, _ = fit(X, y, w, tr_k, va_k, cfg, names, rounds=best)
            oof[va_k] = b.predict_proba(X.iloc[va_k])
            print(f"fold {k} OOF macro-F1 = {metric.macro_f1(y[va_k], oof[va_k].argmax(axis=1)):.4f}")
        cv_rows = np.where(fold >= 0)[0]
        _save_probs(OOF / f"{run_id}.parquet", meta["txn_id"][cv_rows], oof[cv_rows])
        # 3. full fit, a few more rounds because there is 50% more data
        b, _ = fit(X, y, w, np.where(keep)[0], None, cfg, names, rounds=int(best * 1.15) + 1)
        te = np.where(is_test)[0]
        _save_probs(PREDS / f"{run_id}.parquet", meta["txn_id"][te], b.predict_proba(X.iloc[te]))
        log["oof_macro_f1"] = metric.macro_f1(y[cv_rows], oof[cv_rows].argmax(axis=1))
        print(f"all-months OOF macro-F1 = {log['oof_macro_f1']:.4f}")

    log["minutes"] = round((time.time() - t0) / 60, 1)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps(log, indent=2))
    print(f"done in {log['minutes']} min -> runs/{run_id}.json")


if __name__ == "__main__":
    main()
