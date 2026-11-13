"""Pair specialists (the Amazon cross-encoder idea: a dedicated model for the unsure cases).

For each confused pair (A, B): a binary XGBoost (GPU) trained only on rows whose clean label is A or B,
all features. At prediction time it re-decides only rows where the base blend's top-2 classes are A and B.

Validation: specialists fitted on Jul+Aug, base blend = OOF of the best runs, scored on Sep (clean + test-like
noise). Final: refit on Jul-Sep, apply to test rows of the base blend's test predictions.

    python -m src.pair_specialist --name p1            # validate + write submission
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT
from .postprocess import write_submission
from .stack2 import BEST
from .train import load_matrix, train_mask_and_weights
from .train_xgb import PARAMS

PAIRS = [("sim_swap", "account_takeover"), ("phishing_collect", "fake_cashback"), ("mule_account", "layering_chain"),
         ("velocity_bot", "mule_account"), ("refund_abuse", "merchant_collusion"), ("qr_swap", "merchant_collusion")]


def main() -> None:
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="p1")
    ap.add_argument("--config", default="configs/L2_lgbm_fw-clean_b1-f1-beh123-lh2_s0.yaml")
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    meta, X, y, names = load_matrix(cfg)
    keep, _ = train_mask_and_weights(meta, y, cfg)
    ok = meta["keep"].fill_null(True).to_numpy()
    fold = meta["month_fold"].to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    st = meta.join(pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "status"), on="txn_id", how="left", maintain_order="left")["status"].to_numpy()

    def blend(folder):
        fr = [pl.read_parquet(folder / f"{r}.parquet") for r in BEST]
        return fr[0]["txn_id"], np.mean([f.select(LABELS).to_numpy().astype(np.float64) for f in fr], axis=0)
    oid, Po = blend(OOF)
    tid, Pt = blend(PREDS)
    pos = {t: i for i, t in enumerate(meta["txn_id"].to_list())}
    oi = np.array([pos[t] for t in oid.to_list()])
    ti = np.array([pos[t] for t in tid.to_list()])
    sep_mask = fold[oi] == 2

    def apply(P, rows, models):
        pred = P.argmax(1).copy()
        top2 = np.argsort(-P, axis=1)[:, :2]
        n = 0
        for (a, b), m in models.items():
            ia, ib = LABEL_TO_ID[a], LABEL_TO_ID[b]
            sel = np.where(((top2[:, 0] == ia) & (top2[:, 1] == ib)) | ((top2[:, 0] == ib) & (top2[:, 1] == ia)))[0]
            sel = sel[pred[sel] != 0]
            if len(sel) == 0:
                continue
            p = m.predict(xgb.DMatrix(X[rows[sel]], feature_names=names))
            new = np.where(p >= 0.5, ia, ib)
            n += int((new != pred[sel]).sum())
            pred[sel] = new
        return pred, n

    def fit_all(train_rows):
        models = {}
        for a, b in PAIRS:
            ia, ib = LABEL_TO_ID[a], LABEL_TO_ID[b]
            r = train_rows[np.isin(y[train_rows], [ia, ib])]
            yy = (y[r] == ia).astype(int)
            params = {k: v for k, v in PARAMS.items() if k not in ("num_class", "objective", "eval_metric")}
            params["objective"] = "binary:logistic"
            models[(a, b)] = xgb.train(params, xgb.DMatrix(X[r], yy, feature_names=names), 200)
        return models

    tr = np.where(keep & ok & (fold >= 0) & (fold != 2))[0]
    models = fit_all(tr)
    sep_rows = oi[sep_mask]
    Ps = Po[sep_mask]
    okm = ok[sep_rows]
    Ps, sep_rows = Ps[okm], sep_rows[okm]
    p_new, n = apply(Ps, sep_rows, models)
    yt = y[sep_rows]
    base = np.where(st[sep_rows] == "noise", 0, yt)
    raw = pl.read_parquet(ROOT / "cache" / "raw.parquet", columns=["txn_id", "label"])
    hid = pl.read_parquet(ROOT / "labels_clean.parquet").filter(pl.col("status") == "noise").join(raw, on="txn_id")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    mix = np.bincount(hid, minlength=N_CLASSES).astype(float); mix /= mix.sum()
    ys = []
    for s in range(6):
        g = np.random.default_rng(s); q = base.copy(); f = (q == 0) & (g.random(len(q)) < 0.0232)
        q[f] = g.choice(N_CLASSES, size=int(f.sum()), p=mix); ys.append(q)
    for tag, pred in (("best blend", Ps.argmax(1)), ("+ pair specialists", p_new)):
        print(f"{tag:<20} Sep clean {metric.macro_f1(yt, pred):.4f}   test-like {np.mean([metric.macro_f1(q, pred) for q in ys]):.4f}", flush=True)
    print(f"Sep rows re-decided: {n}; {time.time() - t0:.0f}s")
    if args.no_sub:
        return
    models = fit_all(np.where(keep & ok & (fold >= 0))[0])
    lab, n = apply(Pt, ti, models)
    print(f"October rows re-decided: {n}")
    sub = pl.DataFrame({"id": tid, "label": [LABELS[i] for i in lab]})
    write_submission(sub, f"L2_pairspec_{args.name}", {"run": f"pairspec_{args.name}", "pairs": PAIRS,
                                                         "predicted_fraud_share": float((lab != 0).mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
