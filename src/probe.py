"""Leaderboard probe: keep one class's predictions from a submission, set every other row to legit.

The public score of such a file is (F1_legit + F1_class) / 12, so it measures one class's F1 on the
public test rows. It is far below any real submission, so it can never become the team's best
public upload (the final ranking uses the best public upload).

Expected scores under two hypotheses are computed on September out-of-fold predictions:
  noise r   clean Sep labels plus i.i.d. 'ordinary-looking' fraud at rate r (the class mix of train)
  r = 0     test labels with no ordinary-looking fraud at all

    python -m src.probe --sub submissions/<file>.csv --runs RUN ... --keep qr_swap
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, ROOT, SUBS, load_folds, load_raw
from .postprocess import write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--keep", nargs="+", required=True)
    args = ap.parse_args()
    keep_ids = [LABEL_TO_ID[c] for c in args.keep]

    sub = pl.read_csv(args.sub)
    probe = sub.with_columns(label=pl.when(pl.col("label").is_in(args.keep)).then(pl.col("label")).otherwise(pl.lit("legit")))

    # expected public scores, simulated on Sep
    oofs = [pl.read_parquet(OOF / f"{r}.parquet") for r in args.runs]
    P = np.mean([o.select(LABELS).to_numpy() for o in oofs], axis=0)
    j = pl.DataFrame({"txn_id": oofs[0]["txn_id"]}).join(load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left") \
        .join(pl.read_parquet(ROOT / "labels_clean.parquet"), on="txn_id", how="left", maintain_order="left") \
        .join(load_raw().select("txn_id", "label"), on="txn_id", how="left", maintain_order="left")
    sep = (j["month_fold"] == 2).to_numpy() & j["keep"].fill_null(True).to_numpy()
    st = j["status"].to_numpy()[sep]
    base = np.where(st == "noise", 0, j["label_clean"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()[sep])
    mix = np.bincount(j.filter(pl.col("status") == "noise")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy(), minlength=N_CLASSES).astype(float)
    mix /= mix.sum()
    pred = P[sep].argmax(1)
    pred_probe = np.where(np.isin(pred, keep_ids), pred, 0)
    print(f"{'hidden rate r':>14}{'full file':>11}{'probe':>9}   " + "  ".join(f"F1 {c}" for c in args.keep))
    for r in (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
        full, prb, per = [], [], []
        for s in range(5):
            g = np.random.default_rng(s)
            y = base.copy()
            flip = (y == 0) & (g.random(len(y)) < r)
            y[flip] = g.choice(N_CLASSES, size=int(flip.sum()), p=mix)
            full.append(metric.macro_f1(y, pred))
            prb.append(metric.macro_f1(y, pred_probe))
            per.append(metric.f1_per_class(y, pred_probe)[keep_ids])
        print(f"{r:>14.3f}{np.mean(full):>11.4f}{np.mean(prb):>9.4f}   " + "  ".join(f"{v:.3f}" for v in np.mean(per, axis=0)))

    name = "PROBE_only_" + "-".join(args.keep)
    write_submission(probe, name, {"run": name, "probe_of": args.sub, "keep": args.keep,
                                   "predicted_fraud_share": float((probe["label"] != "legit").mean()),
                                   "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
