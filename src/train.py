"""Config-driven trainer. One command per run:

    python -m src.train --config configs/<name>.yaml              # everything
    python -m src.train --config configs/<name>.yaml --mode sep   # only the honest Sep score

What a full run does (research.md D2):
  1. Train Jul+Aug, validate Sep with early stopping on macro-F1  -> the honest score, Sep OOF
  2. Train on the other two months for the Jul fold and the Aug fold -> Jul and Aug OOF
  3. Train on all of Jul-Sep -> test predictions
The Oct-dated train rows are in every training set and never in validation.

Outputs: oof/<run_id>.parquet, preds/<run_id>.parquet (txn_id + 12 probability columns in the
LABELS order) and runs/<run_id>.json (one file per run, so git never conflicts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import metric
from .common import (LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT, RUNS, load_features,
                     load_folds, load_raw)

DEFAULT_PARAMS = {
    "objective": "multiclass", "num_class": N_CLASSES, "metric": "None",
    "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 40,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_bin": 255, "verbose": -1,
}


def load_matrix(cfg: dict):
    """Returns ids/meta frame, feature matrix (float32), labels (-1 for test), feature names."""
    view = cfg.get("view", "half_s0")
    meta = load_raw().select("txn_id", "split", "label", "month_idx").join(
        load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left"
    )
    feats = meta.select("txn_id")
    for group in cfg["features"]:
        f = load_features(group, view)
        clash = (set(f.columns) - {"txn_id"}) & set(feats.columns)
        assert not clash, f"{group}: duplicate feature names {sorted(clash)[:5]}"
        feats = feats.join(f, on="txn_id", how="left", maintain_order="left")
    drop = set(cfg.get("drop_features", []))
    names = [c for c in feats.columns if c != "txn_id" and c not in drop]
    X = feats.select(pl.col(names).cast(pl.Float32)).to_numpy()

    labels = meta["label"]
    if cfg.get("labels", "raw") == "clean":
        clean = pl.read_parquet(ROOT / cfg.get("labels_file", "labels_clean.parquet")).select("txn_id", "label_clean", "keep")
        meta = meta.join(clean, on="txn_id", how="left", maintain_order="left")
        labels = meta["label_clean"].fill_null(meta["label"])
    y = labels.replace_strict(LABEL_TO_ID, default=-1, return_dtype=pl.Int32).to_numpy()
    return meta, X, y, names


def train_mask_and_weights(meta: pl.DataFrame, y: np.ndarray, cfg: dict):
    """Row weights: class weight (n_legit / n_class) ** power, legit rows optionally downsampled."""
    is_train = (meta["split"] == "train").to_numpy()
    counts = np.bincount(y[is_train], minlength=N_CLASSES).astype(np.float64)
    class_w = (counts[0] / np.maximum(counts, 1.0)) ** cfg.get("class_weight_power", 0.5)
    w = np.where(y >= 0, class_w[np.maximum(y, 0)], 0.0)

    keep = is_train.copy()
    if "keep" in meta.columns:  # rows flagged by label cleaning are dropped from training
        keep &= meta["keep"].fill_null(True).to_numpy()
    cfg["_val_weight"] = w.astype(np.float32)  # class weights only, used for early stopping
    frac = cfg.get("legit_keep", 1.0)
    if frac < 1.0:  # deterministic downsampling of legit rows, weight scaled back up
        seed = cfg.get("seed", 0)
        h = np.fromiter(
            (hashlib.md5(f"ds{seed}:{t}".encode()).digest()[0] for t in meta["txn_id"].to_list()),
            dtype=np.uint8, count=meta.height,
        )
        drop = (y == 0) & (h >= int(256 * frac))
        keep &= ~drop
        w = np.where(y == 0, w / frac, w)
    return keep, w.astype(np.float32)


def _feval(preds: np.ndarray, data):
    y = data.get_label().astype(np.int64)
    p = preds.reshape(len(y), -1) if preds.ndim == 1 else preds
    return "macro_f1", metric.macro_f1(y, p.argmax(axis=1)), True


def fit(X, y, w, tr_idx, va_idx, cfg, names, rounds=None):
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **cfg.get("params", {}), "seed": cfg.get("seed", 0)}
    cat = [n for n in names if n.startswith("cat_")]
    dtr = lgb.Dataset(X[tr_idx], y[tr_idx], weight=w[tr_idx], feature_name=names, categorical_feature=cat)
    if rounds is None:  # early stopping on the validation fold
        # 'logloss': class-weighted multi_logloss on the validation fold, the same
        # objective the model trains on. Smooth, so it does not stop on a lucky early round.
        # 'macro_f1' (default for now): plain-argmax macro-F1, the original behaviour (noisy; stopped after ~20 rounds).
        stop_on = cfg.get("early_stop_metric", "macro_f1")
        if stop_on == "logloss":
            params["metric"] = "multi_logloss"
            dva = lgb.Dataset(X[va_idx], y[va_idx], weight=cfg["_val_weight"][va_idx], reference=dtr)
            feval = None
        else:
            dva = lgb.Dataset(X[va_idx], y[va_idx], reference=dtr)
            feval = _feval
        booster = lgb.train(
            params, dtr, cfg.get("num_boost_round", 2000), valid_sets=[dva], valid_names=["sep"],
            feval=feval,
            callbacks=[lgb.early_stopping(cfg.get("early_stopping", 60)), lgb.log_evaluation(100)],
        )
        return booster, booster.best_iteration
    return lgb.train(params, dtr, rounds), rounds


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _save_probs(path: Path, ids: pl.Series, probs: np.ndarray) -> None:
    path.parent.mkdir(exist_ok=True)
    pl.DataFrame({"txn_id": ids}).with_columns(
        [pl.Series(LABELS[c], probs[:, c], dtype=pl.Float32) for c in range(N_CLASSES)]
    ).write_parquet(path)


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
    print(f"run {run_id}: {X.shape[1]} features, {int(keep.sum())} training rows after sampling")

    # 1. honest score: train Jul+Aug (+ Oct-dated train rows), validate Sep
    # With labels: clean, rows marked keep = False (ambiguous or untyped unreported fraud) are
    # predicted but not scored and not used for early stopping. With raw labels nothing changes.
    ok = meta["keep"].fill_null(True).to_numpy() if "keep" in meta.columns else np.ones(meta.height, dtype=bool)
    sep_all = np.where(fold == 2)[0]
    va = np.where((fold == 2) & ok)[0]
    tr = np.where(keep & (fold != 2))[0]
    booster, best = fit(X, y, w, tr, va, cfg, names)
    oof = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    oof[sep_all] = booster.predict(X[sep_all], num_iteration=best)
    rep = metric.report(y[va], oof[va].argmax(axis=1), f"\nSep validation, {best} rounds (plain argmax)")
    imp = sorted(zip(booster.feature_importance("gain"), names), reverse=True)[:25]
    print("top features by gain:", ", ".join(n for _, n in imp))

    log = {
        "run_id": run_id, "config": args.config, "commit": _git_commit(), "features": cfg["features"],
        "view": cfg.get("view", "half_s0"), "seed": cfg.get("seed", 0), "best_rounds": int(best),
        "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"], "n_features": len(names),
    }
    if args.mode == "all":
        # 2. OOF for Jul and Aug, same number of rounds
        for k in (0, 1):
            va_k = np.where(fold == k)[0]
            tr_k = np.where(keep & (fold != k))[0]
            b, _ = fit(X, y, w, tr_k, va_k, cfg, names, rounds=best)
            oof[va_k] = b.predict(X[va_k])
            sc = va_k[ok[va_k]]
            print(f"fold {k} OOF macro-F1 = {metric.macro_f1(y[sc], oof[sc].argmax(axis=1)):.4f}")
        cv_rows = np.where(fold >= 0)[0]
        _save_probs(OOF / f"{run_id}.parquet", meta["txn_id"][cv_rows], oof[cv_rows])
        cv_rows = cv_rows[ok[cv_rows]]  # score only the rows whose label we trust
        # 3. full fit, a few more rounds because there is 50% more data
        b, _ = fit(X, y, w, np.where(keep)[0], None, cfg, names, rounds=int(best * 1.15) + 1)
        te = np.where(is_test)[0]
        _save_probs(PREDS / f"{run_id}.parquet", meta["txn_id"][te], b.predict(X[te]))
        log["oof_macro_f1"] = metric.macro_f1(y[cv_rows], oof[cv_rows].argmax(axis=1))
        print(f"all-months OOF macro-F1 = {log['oof_macro_f1']:.4f}")

    log["minutes"] = round((time.time() - t0) / 60, 1)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps(log, indent=2))
    print(f"done in {log['minutes']} min -> runs/{run_id}.json")


if __name__ == "__main__":
    main()
