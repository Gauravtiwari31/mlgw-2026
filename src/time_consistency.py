"""Time-consistency screen (work/AWS task 4, the IEEE-CIS 1st place check).

Per feature: a one-feature LightGBM (fraud vs legit, clean labels) trained on July, scored by AUC on
July itself and on September. A feature whose September AUC is at or below 0.5, or far below its July
AUC, does not carry the same meaning across months and is a drop candidate. The October column comes
from src.audit_october (anchor-row drift, label-free) when that file exists.

    python -m src.time_consistency --view fw_s0 --groups base_v1 flow_v1 behaviour_v1 behaviour_v2 behaviour_v3 labelhist_v2
Writes audit/time_consistency.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from .common import ROOT, load_features, load_raw


def main() -> None:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="fw_s0")
    ap.add_argument("--groups", nargs="+", default=["base_v1", "flow_v1", "behaviour_v1", "behaviour_v2"])
    ap.add_argument("--legit-frac", type=float, default=0.1)
    args = ap.parse_args()
    t0 = time.time()
    raw = load_raw().select("txn_id", "split", "month_idx")
    cl = pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "label_clean", "keep")
    meta = raw.join(cl, on="txn_id", how="left", maintain_order="left")
    y = (meta["label_clean"] != "legit").fill_null(False).to_numpy().astype(int)
    ok = meta["keep"].fill_null(False).to_numpy() & (meta["split"] == "train").to_numpy()
    m = meta["month_idx"].to_numpy()
    rng = np.random.default_rng(0)
    pick = ok & ((y == 1) | (rng.random(len(y)) < args.legit_frac))
    jul, sep = np.where(pick & (m == 0))[0], np.where(pick & (m == 2))[0]
    rows = []
    for g in args.groups:
        f = load_features(g, args.view)
        for c in [c for c in f.columns if c != "txn_id"]:
            x = f[c].cast(pl.Float32).to_numpy().reshape(-1, 1)
            if np.isnan(x[jul]).all():
                continue
            b = lgb.train({"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 200, "verbose": -1},
                          lgb.Dataset(x[jul], y[jul]), 60)
            a_jul = roc_auc_score(y[jul], b.predict(x[jul]))
            a_sep = roc_auc_score(y[sep], b.predict(x[sep]))
            rows.append({"group": g, "feature": c, "auc_jul": round(a_jul, 4), "auc_sep": round(a_sep, 4), "drop_jul_to_sep": round(a_jul - a_sep, 4)})
        print(f"{g}: {sum(r['group'] == g for r in rows)} features ({time.time() - t0:.0f}s)", flush=True)
    t = pl.DataFrame(rows)
    drift = ROOT / "audit" / f"october_drift__{args.view}.csv"
    if drift.exists():
        t = t.join(pl.read_csv(drift).select("feature", oct_change="change", oct_month_spread="train_month_spread"), on="feature", how="left")
    t = t.with_columns(flag=pl.when(pl.col("auc_sep") <= 0.5).then(pl.lit("no signal in Sep"))
                       .when(pl.col("drop_jul_to_sep") > 0.05).then(pl.lit("weaker in Sep"))
                       .otherwise(pl.lit("")))
    out = ROOT / "audit" / "time_consistency.csv"
    t.sort("drop_jul_to_sep", descending=True).write_csv(out)
    print(t.filter(pl.col("flag") != "").sort("drop_jul_to_sep", descending=True))
    print(f"{t.height} features screened, {t.filter(pl.col('flag') != '').height} flagged -> {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
