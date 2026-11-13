"""Pseudo-labelling for the month shift (XGBoost on the GPU).

Validation (honest): pretend September is the unseen month.
  A: train Jul+Aug, predict Sep.
  B: train Jul+Aug + Sep rows whose A-probability >= --conf (labelled with A's argmax), predict Sep.
  Compare A and B on Sep (clean labels and with simulated test noise).
Final (--final): train Jul-Sep + October test rows with confidence >= --conf from the given runs' blend,
predict October.

    python -m src.pseudo --config configs/L2_lgbm_fw-clean_b1-f1-beh12_s0.yaml --conf 0.97
    python -m src.pseudo --config ... --conf 0.97 --final --run-id <id> --teacher RUN ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, PREDS, ROOT
from .train import _save_probs, load_matrix, train_mask_and_weights
from .train_xgb import PARAMS


def train(X, y, w, idx, names, rounds, seed):
    import xgboost as xgb
    return xgb.train({**PARAMS, "seed": seed}, xgb.QuantileDMatrix(X[idx], y[idx], weight=w[idx], feature_names=names), rounds)


def predict(b, X, idx, names):
    import xgboost as xgb
    return b.predict(xgb.DMatrix(X[idx], feature_names=names))


def noisy_scores(y, pred, status):
    base = np.where(status == "noise", 0, y)
    cl = pl.read_parquet(ROOT / "labels_clean.parquet")
    raw = pl.read_parquet(ROOT / "cache" / "raw.parquet", columns=["txn_id", "label"])
    hid = cl.filter(pl.col("status") == "noise").join(raw, on="txn_id")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    mix = np.bincount(hid, minlength=N_CLASSES).astype(float); mix /= mix.sum()
    out = []
    for s in range(6):
        g = np.random.default_rng(s); yy = base.copy(); f = (yy == 0) & (g.random(len(yy)) < 0.0232)
        yy[f] = g.choice(N_CLASSES, size=int(f.sum()), p=mix); out.append(metric.macro_f1(yy, pred))
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conf", type=float, default=0.97)
    ap.add_argument("--rounds", type=int, default=0, help="0 = best_rounds of the matching xgb run")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--teacher", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    meta, X, y, names = load_matrix(cfg)
    keep, w = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    ok = meta["keep"].fill_null(True).to_numpy()
    rounds = args.rounds or 400
    legit_w = float(np.median(w[keep & (y == 0)]))   # weight a kept legit row carries (scaled for downsampling)
    frac = cfg.get("legit_keep", 1.0)
    rng = np.random.default_rng(args.seed)

    if not args.final:
        tr = np.where(keep & (fold != 2))[0]
        sep = np.where(fold == 2)[0]
        bA = train(X, y, w, tr, names, rounds, args.seed)
        PA = predict(bA, X, sep, names)
        conf, lab = PA.max(1), PA.argmax(1)
        take = conf >= args.conf
        take &= (lab != 0) | (rng.random(len(lab)) < frac)          # same legit downsampling as training
        ps = sep[take]
        y2, w2 = y.copy(), w.copy()
        y2[ps] = lab[take]
        cw = {c: float(np.median(w[keep & (y == c)])) for c in range(N_CLASSES)}
        w2[ps] = np.array([cw[c] for c in lab[take]], dtype=np.float32)
        tr2 = np.concatenate([tr, ps])
        bB = train(X, y2, w2, tr2, names, rounds, args.seed)
        PB = predict(bB, X, sep, names)
        sc = sep[ok[sep]]
        m = np.isin(sep, sc)
        st = meta.join(pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "status"), on="txn_id", how="left", maintain_order="left")["status"].to_numpy()[sc]
        for tag, P in (("A Jul+Aug", PA), ("B + pseudo Sep", PB)):
            pred = P[m].argmax(1)
            print(f"{tag:<16} Sep clean {metric.macro_f1(y[sc], pred):.4f}   Sep + test-like noise {noisy_scores(y[sc], pred, st):.4f}", flush=True)
        print(f"pseudo rows added: {len(ps)} (fraud {int((y2[ps] != 0).sum())}); predictions changed on Sep: {int((PA.argmax(1) != PB.argmax(1)).sum())}; {time.time() - t0:.0f}s")
        return

    te = np.where((meta["split"] == "test").to_numpy())[0]
    fr = [pl.read_parquet(PREDS / f"{r}.parquet") for r in args.teacher]
    Pt = np.mean([f.select(LABELS).to_numpy() for f in fr], axis=0)
    assert fr[0]["txn_id"].equals(meta["txn_id"][te])
    conf, lab = Pt.max(1), Pt.argmax(1)
    take = (conf >= args.conf) & ((lab != 0) | (rng.random(len(lab)) < frac))
    ps = te[take]
    y2, w2 = y.copy(), w.copy()
    y2[ps] = lab[take]
    cw = {c: float(np.median(w[keep & (y == c)])) for c in range(N_CLASSES)}
    w2[ps] = np.array([cw[c] for c in lab[take]], dtype=np.float32)
    tr = np.concatenate([np.where(keep)[0], ps])
    b = train(X, y2, w2, tr, names, int(rounds * 1.15), args.seed)
    P = predict(b, X, te, names)
    _save_probs(PREDS / f"{args.run_id}.parquet", meta["txn_id"][te], P)
    print(f"final: {len(ps)} October pseudo rows (fraud {int((y2[ps] != 0).sum())}); October predictions changed vs teacher: {int((P.argmax(1) != lab).sum())}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
