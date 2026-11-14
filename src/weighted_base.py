"""Blend runs and pick each row's label with fixed class weights on the probabilities.

Why: test labels are clean and most fraud of every class looks ordinary, so macro-F1 rewards a different
cut between look-alike classes than plain argmax. The weights were fitted three times on two train months
and scored on the third (ruler: raw labels, rows flagged unreported left out, phone-id zones applied):
held-out gain +0.0018, +0.0015, +0.0006. Only weights chosen in at least two of the three fits are kept,
at the geometric mean of the three values (audit/device_blocks.md).

    python -m src.weighted_base --name NAME --runs RUN ...       # writes submissions/<stamp>__L2_NAME.csv
Then apply the zones:  python -m src.zone_apply --propagate --base <that file> --runs RUN ...
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import CACHE, LABEL_TO_ID, LABELS, OOF, PREDS, ROOT, load_folds, load_raw
from .postprocess import write_submission

WEIGHTS = {"legit": 0.67, "phishing_collect": 0.37, "fake_cashback": 1.92, "loan_app_extortion": 1.57}


def weight_vector() -> np.ndarray:
    return np.array([WEIGHTS.get(c, 1.0) for c in LABELS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    w = weight_vector()

    # ---- check on train months (weights are fixed, so this is a report, not a fit)
    ref = pl.read_parquet(OOF / f"{args.runs[0]}.parquet").select("txn_id")
    P = np.mean([ref.join(pl.read_parquet(OOF / f"{r}.parquet"), on="txn_id", how="left", maintain_order="left").select(LABELS).to_numpy()
                 for r in args.runs], axis=0)
    m = ref.join(load_raw().select("txn_id", "label"), on="txn_id", how="left", maintain_order="left").join(
        load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left").join(
        pl.read_parquet(CACHE / "device_zone.parquet").select("txn_id", "zone"), on="txn_id", how="left", maintain_order="left").join(
        pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "status"), on="txn_id", how="left", maintain_order="left")
    y = m["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    zid = m["zone"].replace_strict(LABEL_TO_ID, default=-1, return_dtype=pl.Int32).to_numpy()
    fold = m["month_fold"].to_numpy()
    ok = ~(m["status"].is_in(["unreported", "unreported_typed"]).fill_null(False).to_numpy() & (y == 0))
    for f, name in ((0, "Jul"), (1, "Aug"), (2, "Sep")):
        mm = ok & (fold == f)
        a = metric.macro_f1(y[mm], np.where(zid[mm] >= 0, zid[mm], P[mm].argmax(1)))
        b = metric.macro_f1(y[mm], np.where(zid[mm] >= 0, zid[mm], (P[mm] * w).argmax(1)))
        print(f"{name}: plain argmax {a:.4f} -> weighted {b:.4f} ({b - a:+.4f})   (unreported rows left out, zones applied)")

    # ---- test
    ids = load_raw().filter(pl.col("split") == "test").select("txn_id")
    T = np.mean([ids.join(pl.read_parquet(PREDS / f"{r}.parquet"), on="txn_id", how="left", maintain_order="left").select(LABELS).to_numpy()
                 for r in args.runs], axis=0)
    p0, p1 = T.argmax(1), (T * w).argmax(1)
    ch = pl.DataFrame({"was": [LABELS[i] for i in p0[p0 != p1]], "now": [LABELS[i] for i in p1[p0 != p1]]})
    print(f"test: {int((p0 != p1).sum())} of {len(p0)} rows change:", ch.group_by("was", "now").agg(n=pl.len()).sort("n", descending=True).rows())
    sub = pl.DataFrame({"id": ids["txn_id"], "label": [LABELS[i] for i in p1]})
    write_submission(sub, f"L2_{args.name}", {"run": f"weighted_{args.name}", "runs": args.runs, "weights": WEIGHTS,
                                              "predicted_fraud_share": float((p1 != 0).mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
