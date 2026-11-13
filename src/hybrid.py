"""Per-class hybrid of two blends: take chosen classes' probabilities from blend B, the rest from blend A,
then renormalise and take the argmax.

    python -m src.hybrid --name <name> --a RUN ... --b RUN ... --b-classes merchant_collusion refund_abuse
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import LABELS, PREDS
from .postprocess import write_submission


def load(runs):
    fr = [pl.read_parquet(PREDS / f"{r}.parquet") for r in runs]
    for f in fr:
        assert f["txn_id"].equals(fr[0]["txn_id"])
    return fr[0]["txn_id"], np.mean([f.select(LABELS).to_numpy().astype(np.float64) for f in fr], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--b-classes", nargs="+", required=True)
    args = ap.parse_args()
    ids, A = load(args.a)
    ids_b, B = load(args.b)
    assert ids.equals(ids_b)
    cols = [LABELS.index(c) for c in args.b_classes]
    H = A.copy()
    H[:, cols] = B[:, cols]
    H /= H.sum(axis=1, keepdims=True)
    lab = H.argmax(1)
    print(f"rows changed against blend A: {int((lab != A.argmax(1)).sum())}")
    sub = pl.DataFrame({"id": ids, "label": [LABELS[i] for i in lab]})
    write_submission(sub, args.name, {"run": args.name, "a": args.a, "b": args.b, "b_classes": args.b_classes,
                                      "predicted_fraud_share": float((lab != 0).mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
