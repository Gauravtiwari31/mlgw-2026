"""Stacker over run FAMILIES (Amazon lesson: a stacker over all model scores makes the final call).

Each family = the average probabilities of its runs (seeds of one recipe). Inputs to the stacker:
log-probabilities of every family (12 classes each) + a few row-context columns, so it can learn
WHEN to trust which view (e.g. the half-density family for collusion).
Shallow XGBoost on the GPU, clean labels, same class weights as the base models.

Honest score: fit on Jul+Aug OOF rows, score Sep (clean labels and test-like noise), against the
mean blend of the current best. Final: refit on Jul-Sep OOF, apply to the test predictions.

    python -m src.stack2 --name s1            # validate + write submission
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT, load_folds, load_raw
from .postprocess import write_submission

FAMILIES = {
    "fw_lgbm": ["L2_lgbm_fw-clean_b1-f1-beh12_s0", "L2_lgbm_fw-clean_b1-f1-beh12_s1", "L2_lgbm_fw-clean_b1-f1-beh12_s2", "L2_lgbm_fw1-clean_b1-f1-beh12_s3"],
    "fw_xgb": ["L2_xgb_fw-clean_b1-f1-beh12_s0", "L2_xgb_fw-clean_b1-f1-beh12_s1", "L2_xgb_fw-clean_b1-f1-beh12_s2"],
    "beh3_xgb": ["L2_xgb_fw-clean_b1-f1-beh123-lh2_s0", "L2_xgb_fw-clean_b1-f1-beh123-lh2_s1", "L2_xgb_fw-clean_b1-f1-beh123-lh2_s2"],
    "tc_lgbm": ["L2_lgbm_fw-clean_b1-f1-beh12_tc_s0", "L2_lgbm_fw-clean_b1-f1-beh12_tc_s1", "L2_lgbm_fw-clean_b1-f1-beh12_tc_s2", "L2_lgbm_fw1-clean_b1-f1-beh12_tc_s3"],
    "tc_xgb": ["L2_xgb_fw-clean_b1-f1-beh12_tc_s0", "L2_xgb_fw-clean_b1-f1-beh12_tc_s1", "L2_xgb_fw-clean_b1-f1-beh12_tc_s2",
               "L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s0", "L2_xgb_fw-clean_b1-f1-beh123-lh2_tc_s1"],
    "half": ["L2_lgbm_half-clean_b1-f1-beh12_s0", "L2_lgbm_half-clean_b1-f1-beh12_s1", "L2_lgbm_half-clean_b1-f1-beh12_s2", "L2_lgbm_half1-clean_b1-f1-beh12_s3", "L2_xgb_half-clean_b1-f1-beh12_s0"],
    "full": ["L2_lgbm_full-clean_b1-f1-beh12_s0", "L2_lgbm_full-clean_b1-f1-beh12_s1", "L2_lgbm_full-clean_b1-f1-beh12_s2", "L2_lgbm_full-clean_b1-f1-beh12_s3", "L2_xgb_full-clean_b1-f1-beh12_s0"],
    "raw": ["L2_lgbm_fwraw_b1-f1-beh12_s0", "L2_lgbm_fwraw_b1-f1-beh12_s1", "L2_lgbm_fwraw_b1-f1-beh12_s2", "L2_lgbm_fw1raw_b1-f1-beh12_s3", "L2_xgb_fwraw_b1-f1-beh12_s0"],
    "nowt": ["L2_lgbm_fwnowt_b1-f1-beh12_s0", "L2_lgbm_fwnowt_b1-f1-beh12_s1", "L2_lgbm_fwnowt_b1-f1-beh12_s2", "L2_lgbm_fw1nowt_b1-f1-beh12_s3", "L2_xgb_fwnowt_b1-f1-beh12_s0"],
    "typerule": ["L2_lgbm_fwt3_b1-f1-beh12_s0", "L2_lgbm_fwt3_b1-f1-beh12_s1", "L2_lgbm_fw1t3_b1-f1-beh12_s3"],
    "catboost": ["L2_catboost_fw-clean_b1-f1-beh12_s0"],
    "stage2": ["L2_xgb_fw-clean_b1-f1-beh12-s2_s0"],
}
BEST = FAMILIES["fw_lgbm"] + ["L2_xgb_fw-clean_b1-f1-beh12_s0"] + FAMILIES["beh3_xgb"] + FAMILIES["stage2"]   # the 11:50 PM blend


def family_probs(folder, families):
    ids, out = None, {}
    for name, runs in families.items():
        fr = [pl.read_parquet(folder / f"{r}.parquet") for r in runs]
        if ids is None:
            ids = fr[0]["txn_id"]
        for f in fr:
            assert f["txn_id"].equals(ids), name
        out[name] = np.mean([f.select(LABELS).to_numpy().astype(np.float64) for f in fr], axis=0)
    return ids, out


def design(ids, fam, ctx):
    cols = [np.log(np.clip(P, 1e-6, 1.0)) for P in fam.values()]
    names = [f"{n}_{c}" for n in fam for c in LABELS]
    c = pl.DataFrame({"txn_id": ids}).join(ctx, on="txn_id", how="left", maintain_order="left").drop("txn_id")
    cols.append(c.to_numpy().astype(np.float64))
    names += c.columns
    return np.hstack(cols).astype(np.float32), names


def main() -> None:
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="s1")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    raw = load_raw()
    ctx = raw.select("txn_id", is_p2p=(pl.col("receiver_type") == "P2P").cast(pl.Int8),
                     ttype=pl.col("txn_type").rank("dense").cast(pl.Int16), amount_log=pl.col("amount").log1p(),
                     whole=((pl.col("amount") * 100).round(0) % 100 == 0).cast(pl.Int8), hour=pl.col("ts").dt.hour())
    ids, fo = family_probs(OOF, FAMILIES)
    tid, ft = family_probs(PREDS, FAMILIES)
    X, names = design(ids, fo, ctx)
    Xt, _ = design(tid, ft, ctx)
    j = pl.DataFrame({"txn_id": ids}).join(load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left") \
        .join(pl.read_parquet(ROOT / "labels_clean.parquet"), on="txn_id", how="left", maintain_order="left") \
        .join(raw.select("txn_id", "label"), on="txn_id", how="left", maintain_order="left")
    keep = j["keep"].fill_null(True).to_numpy()
    y = j["label_clean"].fill_null(j["label"]).replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    fold = j["month_fold"].to_numpy()
    st = j["status"].to_numpy()
    counts = np.bincount(y[keep], minlength=N_CLASSES).astype(float)
    cw = (counts[0] / np.maximum(counts, 1)) ** 0.5
    rng = np.random.default_rng(0)
    use = keep & ((y != 0) | (rng.random(len(y)) < 0.25))
    w = np.where(y == 0, cw[0] / 0.25, cw[np.maximum(y, 0)]).astype(np.float32)
    params = {"objective": "multi:softprob", "num_class": N_CLASSES, "tree_method": "hist", "device": "cuda",
              "learning_rate": 0.05, "max_depth": args.depth, "min_child_weight": 10, "subsample": 0.8,
              "colsample_bytree": 0.6, "lambda": 10.0, "seed": 0}

    def fit(idx):
        return xgb.train(params, xgb.QuantileDMatrix(X[idx], y[idx], weight=w[idx], feature_names=names), args.rounds)

    tr = np.where(use & (fold != 2))[0]
    sep = np.where(keep & (fold == 2))[0]
    b = fit(tr)
    Ps = b.predict(xgb.DMatrix(X[sep], feature_names=names))
    # baseline: mean blend of BEST on the same rows
    best_o = np.mean([pl.read_parquet(OOF / f"{r}.parquet").select(LABELS).to_numpy() for r in BEST], axis=0)
    Pb = best_o[sep]
    base = np.where(st[sep] == "noise", 0, y[sep])
    hid = j.filter(pl.col("status") == "noise")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    mix = np.bincount(hid, minlength=N_CLASSES).astype(float); mix /= mix.sum()
    ys = []
    for s in range(6):
        g = np.random.default_rng(s); yy = base.copy(); f = (yy == 0) & (g.random(len(yy)) < 0.0232)
        yy[f] = g.choice(N_CLASSES, size=int(f.sum()), p=mix); ys.append(yy)
    for tag, P in (("best mean blend (11:50 PM file)", Pb), (f"stacker depth {args.depth}", Ps)):
        pred = P.argmax(1)
        print(f"{tag:<34} Sep clean {metric.macro_f1(y[sep], pred):.4f}   test-like {np.mean([metric.macro_f1(yy, pred) for yy in ys]):.4f}", flush=True)
    f0 = np.mean([metric.f1_per_class(yy, Pb.argmax(1)) for yy in ys], axis=0)
    f1 = np.mean([metric.f1_per_class(yy, Ps.argmax(1)) for yy in ys], axis=0)
    print("per class (test-like):", {LABELS[c]: round(float(f1[c] - f0[c]), 3) for c in range(N_CLASSES) if abs(f1[c] - f0[c]) >= 0.002})
    imp = sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:12]
    print("top stacker inputs:", ", ".join(k for k, _ in imp))
    if args.no_sub:
        return
    bf = fit(np.where(use)[0])
    Pt = bf.predict(xgb.DMatrix(Xt, feature_names=names))
    lab = Pt.argmax(1)
    best_t = np.mean([pl.read_parquet(PREDS / f"{r}.parquet").select(LABELS).to_numpy() for r in BEST], axis=0).argmax(1)
    print(f"October: {int((lab != best_t).sum())} rows differ from the best blend; fraud share {(lab != 0).mean():.4f}; {time.time() - t0:.0f}s")
    sub = pl.DataFrame({"id": tid, "label": [LABELS[i] for i in lab]})
    write_submission(sub, f"L2_stack2_{args.name}", {"run": f"stack2_{args.name}", "families": list(FAMILIES), "depth": args.depth,
                                                     "predicted_fraud_share": float((lab != 0).mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
