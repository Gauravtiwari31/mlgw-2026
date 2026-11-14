"""Recall-injection layer: class-specific detectors that rescue fraud the blend calls legit.

The layer audit (14 Nov): features cover every fraud definition (flow/behaviour/geography/entity
history), and the 13-run blend catches ~91% of visible fraud on clean Sep. What escapes are
borderline rows predicted `legit` with small margins. This module adds the missing detection
channel: a one-vs-legit LightGBM per weak class, trained on clean labels, whose high-confidence
calls on blend-legit rows are injected back into the predictions.

Honesty rules (same as everywhere else):
- The flip fraction q_c is chosen ONLY on the Jul and Aug OOF folds (detectors trained without
  the fold's month); Sep is then reported untouched as the honest number.
- The test application uses detectors retrained on all train months, with the q_c from Jul+Aug.
- Only rows the BLEND calls legit are eligible for injection; every other prediction is untouched.

    python -m src.recall_inject --runs RUN1 RUN2 ...            # report only
    python -m src.recall_inject --runs ... --write L3_blend_all13
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from . import metric
from .common import LABELS, N_CLASSES, ROOT
from .postprocess import write_submission

TARGETS = ["sim_swap", "account_takeover", "fake_cashback", "velocity_bot",
           "mule_account", "merchant_collusion"]
Q_GRID = [0.0005, 0.001, 0.002, 0.004, 0.007, 0.012]  # fraction of blend-legit rows to flip per class


def load_blend(runs: list[str]):
    """Mean OOF + test probabilities across runs; returns meta info and matrices."""
    oofs = [pl.read_parquet(ROOT / "oof" / f"{r}.parquet") for r in runs]
    preds = [pl.read_parquet(ROOT / "preds" / f"{r}.parquet") for r in runs]
    ids = oofs[0]["txn_id"]
    O = np.mean([o.select(LABELS).to_numpy() for o in oofs], axis=0)
    P = np.mean([p.select(LABELS).to_numpy() for p in preds], axis=0)
    return ids, O, P


def fit_detector(X, y, w, tr_idx, seed):
    """LightGBM one-vs-legit on rows whose label is the target class or legit."""
    import lightgbm as lgb

    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.1,
        "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.7,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "scale_pos_weight": float(max(1.0, (y[tr_idx] == 0).sum() / max(1, (y[tr_idx] == 1).sum()))),
        "seed": seed, "verbose": -1,
    }
    dtr = lgb.Dataset(X[tr_idx], y[tr_idx], weight=w[tr_idx])
    dva = None
    booster = lgb.train(params, dtr, 500)
    return booster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--write", default=None, help="base submission name to inject into")
    args = ap.parse_args()

    from .train import load_matrix, train_mask_and_weights

    cfg = {
        "features": ["base_v1", "flow_v1", "behaviour_v1", "behaviour_v2", "behaviour_v3",
                     "behaviour_v4", "labelhist_v2"],
        "view": "fw_s0", "labels": "clean", "drop_features": ["ip_risk_rank"],
        "seed": 0, "class_weight_power": 0.5, "legit_keep": 0.2,
    }
    meta, X, y, names = load_matrix(cfg)
    keep, w = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    ok = meta["keep"].fill_null(True).to_numpy() if "keep" in meta.columns else np.ones(meta.height, dtype=bool)
    is_test = (meta["split"] == "test").to_numpy()

    ids, O, P = load_blend(args.runs)
    cv_rows = np.where(fold >= 0)[0]
    assert len(ids) == len(cv_rows), "blend oof must cover exactly the cv rows"
    id_list = ids.to_list()
    txn_list = meta["txn_id"].to_list()
    assert id_list[0] == txn_list[cv_rows[0]] and id_list[-1] == txn_list[cv_rows[-1]]
    O_full = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    O_full[cv_rows] = O
    te_rows = np.where(is_test)[0]
    P_full = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    P_full[te_rows] = P

    base_pred = O_full.argmax(axis=1)
    best_q: dict[str, float] = {}
    print("=== choosing flip fractions on Jul+Aug OOF (honest), reporting Sep untouched ===")
    for cls in TARGETS:
        ci = LABELS.index(cls)
        yi = ((y == ci) | (y == 0)).astype(np.int8)  # one-vs-legit: only these rows train
        rows_cls = yi.astype(bool)
        gains = {}
        scores = {}
        for k in (0, 1):
            va = np.where((fold == k) & ok)[0]
            tr = np.where(keep & ok & (fold != k) & rows_cls)[0]
            det = fit_detector(X, (y == ci).astype(np.int32), w, tr, seed=k)
            elig = va[base_pred[va] == 0]  # blend calls legit
            scores[k] = (det.predict(X[elig]), elig)
        for q in Q_GRID:
            pr = base_pred.copy()
            for k in (0, 1):
                s, elig = scores[k]
                n_flip = max(0, int(len(s) * q))
                if n_flip == 0:
                    continue
                top = np.argsort(-s)[:n_flip]
                pr[elig[top]] = ci
            va01 = np.where(((fold == 0) | (fold == 1)) & ok)[0]
            gains[q] = metric.macro_f1(y[va01], pr[va01]) - metric.macro_f1(y[va01], base_pred[va01])
        q_star = max(gains, key=gains.get)
        best_q[cls] = q_star if gains[q_star] > 0.0005 else 0.0
        print(f"{cls:<20} best q={q_star} gain(Jul+Aug)={gains[q_star]:+.4f} "
              f"(all: {' '.join(f'{q}:{g:+.4f}' for q, g in gains.items())})")

    # honest Sep evaluation with the Jul+Aug-chosen q's
    va_sep = np.where((fold == 2) & ok)[0]
    before = metric.macro_f1(y[va_sep], base_pred[va_sep])
    pr_sep = base_pred.copy()
    for cls in TARGETS:
        if best_q[cls] <= 0:
            continue
        ci = LABELS.index(cls)
        rows_cls = ((y == ci) | (y == 0)).astype(bool).astype(np.int8)
        tr = np.where(keep & ok & (fold != 2) & rows_cls)[0]
        det = fit_detector(X, (y == ci).astype(np.int32), w, tr, seed=7)
        elig = va_sep[base_pred[va_sep] == 0]
        s = det.predict(X[elig])
        n_flip = int(len(s) * best_q[cls])
        top = np.argsort(-s)[:n_flip]
        pr_sep[elig[top]] = ci
        n_true = int((y[elig[top]] == ci).sum())
        print(f"Sep {cls:<20}: {n_flip} flips, {n_true} true ({100.0 * n_true / max(1, n_flip):.0f}% precision)")
    after = metric.macro_f1(y[va_sep], pr_sep[va_sep])
    rep = metric.report(y[va_sep], pr_sep[va_sep], "\nSep after recall injection")
    print(f"Sep macro-F1: {before:.4f} -> {after:.4f} ({after - before:+.4f})")

    if args.write and after > before + 0.002:
        test_pred = P_full.argmax(axis=1)
        for cls in TARGETS:
            if best_q[cls] <= 0:
                continue
            ci = LABELS.index(cls)
            rows_cls = ((y == ci) | (y == 0)).astype(bool).astype(np.int8)
            tr = np.where(keep & ok & rows_cls)[0]  # all train months for the test application
            det = fit_detector(X, (y == ci).astype(np.int32), w, tr, seed=13)
            elig = te_rows[test_pred == 0]
            s = det.predict(X[elig])
            n_flip = int(len(s) * best_q[cls])
            top = np.argsort(-s)[:n_flip]
            test_pred[elig[top]] = ci
            print(f"test {cls:<20}: {n_flip} rows injected")
        sub = pl.DataFrame({
            "id": meta["txn_id"][te_rows],
            "label": [LABELS[i] for i in test_pred],
        })
        write_submission(sub, args.write + "_recall", {"base": args.write, "sep_gain": round(after - before, 4)})
    elif args.write:
        print("no Sep gain above +0.002; submission not written")


if __name__ == "__main__":
    main()
