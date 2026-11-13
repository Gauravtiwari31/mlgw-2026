"""Turn probabilities into labels for macro-F1 (research.md D6).

Weighted argmax: predict argmax(w_c * p_c). The 12 weights are found by coordinate ascent on
out-of-fold probabilities. They are fitted on Jul+Aug and checked on Sep, so the reported gain
is honest; the weights used for the submission are then refitted on all three months.

    python -m src.postprocess --run <run_id>            # report + write submissions/<run_id>.csv
    python -m src.postprocess --run <run_id> --no-sub   # report only
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, SUBS, load_folds, load_raw

GRID = np.exp(np.linspace(-3.0, 3.0, 25))


def fit_weights(P: np.ndarray, y: np.ndarray, n_pass: int = 3) -> tuple[np.ndarray, float]:
    """Greedy coordinate ascent over one class weight at a time."""
    w = np.ones(N_CLASSES)
    best = metric.macro_f1(y, (P * w).argmax(axis=1))
    for _ in range(n_pass):
        improved = False
        for c in range(N_CLASSES):
            for g in GRID:
                trial = w.copy()
                trial[c] = g
                score = metric.macro_f1(y, (P * trial).argmax(axis=1))
                if score > best + 1e-6:
                    best, w, improved = score, trial, True
        if not improved:
            break
    return w / w[0], best


def predict(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (P * w).argmax(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()

    oof = pl.read_parquet(OOF / f"{args.run}.parquet")
    truth = load_raw().select("txn_id", "label").join(load_folds().select("txn_id", "month_fold"), on="txn_id")
    oof = oof.join(truth, on="txn_id", how="left", maintain_order="left")
    P = oof.select(LABELS).to_numpy().astype(np.float64)
    y = oof["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    fold = oof["month_fold"].to_numpy()
    sep, rest = fold == 2, fold != 2

    before = metric.macro_f1(y[sep], P[sep].argmax(axis=1))
    w_fit, _ = fit_weights(P[rest], y[rest])
    metric.report(y[sep], predict(P[sep], w_fit), "\nSep, weights fitted on Jul+Aug only")
    after = metric.macro_f1(y[sep], predict(P[sep], w_fit))
    print(f"Sep macro-F1: plain argmax {before:.4f} -> weighted argmax {after:.4f}")

    w_all, score_all = fit_weights(P, y)
    print("weights refitted on Jul-Sep:", {LABELS[c]: round(float(w_all[c]), 2) for c in range(N_CLASSES)})
    print(f"in-sample macro-F1 with those weights: {score_all:.4f} (optimistic, do not quote)")

    if not args.no_sub:
        preds = pl.read_parquet(PREDS / f"{args.run}.parquet")
        lab = predict(preds.select(LABELS).to_numpy().astype(np.float64), w_all)
        sub = pl.DataFrame({"id": preds["txn_id"], "label": [LABELS[i] for i in lab]})
        assert sub.height == load_raw().filter(pl.col("split") == "test").height
        SUBS.mkdir(exist_ok=True)
        path = SUBS / f"{args.run}.csv"
        sub.write_csv(path)
        share = sub.group_by("label").agg(rows=pl.len()).sort("rows", descending=True)
        print(f"wrote {path.name}; predicted label counts:\n{share}")


if __name__ == "__main__":
    main()
