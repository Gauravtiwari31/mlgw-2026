"""F1-optimal decisions under noisy test labels (Lipton et al. 2014: the optimal cut-off for a calibrated
probability is half the best achievable F1).

1. Average the runs' probabilities; calibrate them on out-of-fold rows against clean labels
   (src.postprocess.fit_calibration), so q_c = P(clean label c | row).
2. Per fraud class c, predict c when q_c is the largest fraud probability and q_c >= t_c, else legit.
   t_c = F_c / 2, where F_c is the class's F1 measured on the public board (one-class probes) or an
   estimate for classes without a probe.
3. Report: rows changed in October, and September under simulated test noise (which overstates F,
   so it understates the gain).

    python -m src.decide --name <name> --runs RUN ...
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT, load_folds, load_raw
from .postprocess import calibrate, fit_calibration, write_submission

# F1 on the public board from one-class probes (13 Nov); other classes: September simulation under noise
F_BOARD = {"qr_swap": 0.53, "mule_account": 0.50, "velocity_bot": 0.35, "merchant_collusion": 0.27}
F_SIM = {"layering_chain": 0.58, "sim_swap": 0.40, "account_takeover": 0.41, "phishing_collect": 0.53,
         "refund_abuse": 0.58, "fake_cashback": 0.42, "loan_app_extortion": 0.44}


def decide(Q: np.ndarray, t: np.ndarray) -> np.ndarray:
    fraud = Q[:, 1:]
    c = fraud.argmax(axis=1)
    best = fraud[np.arange(len(Q)), c]
    return np.where(best >= t[c + 1], c + 1, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all cut-offs (1.0 = F/2)")
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()
    oofs = [pl.read_parquet(OOF / f"{r}.parquet") for r in args.runs]
    tests = [pl.read_parquet(PREDS / f"{r}.parquet") for r in args.runs]
    P = np.mean([o.select(LABELS).to_numpy().astype(np.float64) for o in oofs], axis=0)
    Pt = np.mean([t.select(LABELS).to_numpy().astype(np.float64) for t in tests], axis=0)
    j = pl.DataFrame({"txn_id": oofs[0]["txn_id"]}).join(load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left") \
        .join(pl.read_parquet(ROOT / "labels_clean.parquet"), on="txn_id", how="left", maintain_order="left") \
        .join(load_raw().select("txn_id", "label"), on="txn_id", how="left", maintain_order="left")
    keep = j["keep"].fill_null(True).to_numpy()
    y = j["label_clean"].fill_null(j["label"]).replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    fold = j["month_fold"].to_numpy()
    rest, sep = keep & (fold != 2), keep & (fold == 2)
    cal = fit_calibration(P[rest], y[rest])          # honest for Sep
    cal_all = fit_calibration(P[keep], y[keep])      # for the test month
    Q, Qt = calibrate(P, cal), calibrate(Pt, cal_all)
    F = {**F_SIM, **F_BOARD}
    t = np.array([1.0] + [F[c] / 2 * args.scale for c in LABELS[1:]])
    print("cut-offs:", {LABELS[c]: round(float(t[c]), 3) for c in range(1, N_CLASSES)})

    # Sep under simulated test noise
    st = j["status"].to_numpy()[sep]
    base = np.where(st == "noise", 0, y[sep])
    mix = np.bincount(j.filter(pl.col("status") == "noise")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy(), minlength=N_CLASSES).astype(float)
    mix /= mix.sum()
    ys = []
    for s in range(6):
        g = np.random.default_rng(s); yy = base.copy(); f = (yy == 0) & (g.random(len(yy)) < 0.0232)
        yy[f] = g.choice(N_CLASSES, size=int(f.sum()), p=mix); ys.append(yy)
    p0, p1 = P[sep].argmax(1), decide(Q[sep], t)
    f0 = np.mean([metric.f1_per_class(yy, p0) for yy in ys], axis=0)
    f1 = np.mean([metric.f1_per_class(yy, p1) for yy in ys], axis=0)
    print(f"Sep + noise: argmax {f0.mean():.4f} -> cut-offs {f1.mean():.4f} ({f1.mean() - f0.mean():+.4f})")
    print(pl.DataFrame({"class": LABELS, "argmax": f0.round(3), "cutoff": f1.round(3), "diff": (f1 - f0).round(3),
                        "sep_pred_argmax": np.bincount(p0, minlength=N_CLASSES), "sep_pred_cut": np.bincount(p1, minlength=N_CLASSES)}))
    l0, l1 = Pt.argmax(1), decide(Qt, t)
    print(f"October: {int((l0 != l1).sum())} rows change; fraud share {(l0 != 0).mean():.4f} -> {(l1 != 0).mean():.4f}")
    ch = pl.DataFrame({"old": [LABELS[i] for i in l0], "new": [LABELS[i] for i in l1]}).filter(pl.col("old") != pl.col("new"))
    print(ch.group_by("old", "new").agg(n=pl.len()).sort("n", descending=True).head(15))
    if args.no_sub:
        return
    sub = pl.DataFrame({"id": tests[0]["txn_id"], "label": [LABELS[i] for i in l1]})
    write_submission(sub, args.name, {"run": args.name, "runs": args.runs, "cutoffs": dict(zip(LABELS, map(float, t))),
                                      "scale": args.scale, "sep_macro_f1_honest": float(f1.mean()), "predicted_fraud_share": float((l1 != 0).mean())})


if __name__ == "__main__":
    main()
