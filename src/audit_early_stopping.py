"""Audit: is early stopping on plain-argmax macro-F1 leaving score on the table?

For each variant, train Jul+Aug for a fixed number of rounds and look at Sep at several
checkpoints:
  plain     macro-F1 of the plain argmax (what src.train stops on)
  weighted  macro-F1 after the weighted argmax, cross-fitted inside Sep: weights are fitted on
            one random half of Sep and scored on the other half, both ways, then averaged.
            This is an honest estimate of what src.postprocess can add at that checkpoint.
  logloss   multi-class log loss on Sep (unweighted)

    python -m src.audit_early_stopping --config configs/L2_lgbm_base1-flow1_s0.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from . import metric
from .common import N_CLASSES, ROOT
from .postprocess import fit_weights, predict
from .train import DEFAULT_PARAMS, load_matrix, train_mask_and_weights

CHECKPOINTS = [10, 20, 30, 50, 80, 120, 180, 250, 350, 500]
VARIANTS = {  # name -> overrides: top-level config keys, or params.* for LightGBM parameters
    "current": {},
    "lr 0.03": {"params.learning_rate": 0.03},
    "smaller trees": {"params.num_leaves": 15, "params.min_data_in_leaf": 200},
    "smaller trees, lr 0.03": {"params.num_leaves": 15, "params.min_data_in_leaf": 200, "params.learning_rate": 0.03},
    "no class weights": {"class_weight_power": 0.0},
    "no class weights, smaller trees, lr 0.03": {
        "class_weight_power": 0.0, "params.num_leaves": 15, "params.min_data_in_leaf": 200,
        "params.learning_rate": 0.03},
    "all legit rows": {"legit_keep": 1.0},
}


def apply(cfg: dict, overrides: dict) -> dict:
    out = {**cfg, "params": dict(cfg.get("params", {}))}
    for k, v in overrides.items():
        if k.startswith("params."):
            out["params"][k[7:]] = v
        else:
            out[k] = v
    return out


def logloss(y: np.ndarray, P: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0)).mean())


def crossfit_weighted(y: np.ndarray, P: np.ndarray, half: np.ndarray) -> float:
    scores = []
    for a in (0, 1):
        w, _ = fit_weights(P[half == a], y[half == a])
        scores.append(metric.macro_f1(y[half != a], predict(P[half != a], w)))
    return float(np.mean(scores))


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--rounds", type=int, default=CHECKPOINTS[-1])
    args = ap.parse_args()
    cfg0 = yaml.safe_load(Path(args.config).read_text())
    meta, X, y, names = load_matrix(cfg0)
    fold = meta["month_fold"].to_numpy()
    va = np.where(fold == 2)[0]
    yv = y[va].astype(np.int64)
    # a fixed random split of the Sep rows, from the txn_id hash
    half = np.fromiter((hashlib.md5(f"es:{t}".encode()).digest()[0] & 1 for t in meta["txn_id"][va].to_list()),
                       dtype=np.int8, count=len(va))
    cat = [n for n in names if n.startswith("cat_")]
    cps = [c for c in CHECKPOINTS if c <= args.rounds]

    out = {"config": Path(args.config).as_posix(), "checkpoints": cps, "variants": {}}
    path = ROOT / "audit" / "early_stopping.json"
    path.parent.mkdir(exist_ok=True)
    for name in args.variants:
        cfg = apply(cfg0, VARIANTS[name])
        keep, w = train_mask_and_weights(meta, y, cfg)
        tr = np.where(keep & (fold != 2))[0]
        params = {**DEFAULT_PARAMS, **cfg.get("params", {}), "seed": cfg.get("seed", 0)}
        t0 = time.time()
        booster = lgb.train(
            params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=names, categorical_feature=cat),
            cps[-1])
        rows = []
        for k in cps:
            P = booster.predict(X[va], num_iteration=k).astype(np.float64)
            pred = P.argmax(axis=1)
            rows.append({
                "rounds": k, "plain": metric.macro_f1(yv, pred), "weighted": crossfit_weighted(yv, P, half),
                "logloss": logloss(yv, P), "pred_fraud_share": float((pred != 0).mean()),
            })
            r = rows[-1]
            print(f"{name:<42} rounds {k:>4}  plain {r['plain']:.4f}  weighted {r['weighted']:.4f}  "
                  f"logloss {r['logloss']:.4f}  pred fraud {r['pred_fraud_share']:.2%}", flush=True)
        out["variants"][name] = {"overrides": VARIANTS[name], "rows": rows, "minutes": round((time.time() - t0) / 60, 1)}
        path.write_text(json.dumps(out, indent=2))
    print(f"true fraud share in Sep: {(yv != 0).mean():.2%}  ->  {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
