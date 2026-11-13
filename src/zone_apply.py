"""Apply device-id zones (src/device_block.py) to predictions: every row on a block phone gets its zone's class.

    python -m src.zone_apply --base submissions/<file>.csv --runs RUN ...   # RUN = the blend behind the base file

Prints the September effect (out-of-fold predictions, raw labels, zones recovered without labels) and writes
the base file with the test rows on block phones overridden.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import CACHE, LABEL_TO_ID, LABELS, N_CLASSES, OOF, load_folds, load_raw
from .postprocess import write_submission


FANIN = ["mule_account", "velocity_bot", "layering_chain"]
TAKEOVER = ["sim_swap", "account_takeover"]


def propagate(rows: pl.DataFrame) -> pl.DataFrame:
    """rows: txn_id, t (epoch s), sender_id, receiver_id, is_p2p, pred (class name), zone (null off the block).
    Returns txn_id, prop for off-block rows whose class is settled by a block row of the same episode:
      R1  same personal receiver within 24 h, both the row's prediction and the zone among mule / velocity / layering
      R2  same sender within 2 h, both among sim_swap / account_takeover
    (train check, audit/device_blocks.md: zone right in 98% and 100% of disagreements; other links are not used)."""
    blk = rows.filter(pl.col("zone").is_not_null())
    rest = rows.filter(pl.col("zone").is_null())
    out = []
    for key, win, group, cond in (("receiver_id", 86400, FANIN, pl.col("is_p2p")), ("sender_id", 7200, TAKEOVER, pl.lit(True))):
        r = rest.filter(cond & pl.col("pred").is_in(group)).select("txn_id", key, "t", "pred")
        b = blk.filter(cond & pl.col("zone").is_in(group)).select(key, bt="t", bz="zone")
        j = r.join(b, on=key, how="inner").filter((pl.col("t") - pl.col("bt")).abs() <= win)
        out.append(j.group_by("txn_id").agg(prop=pl.col("bz").mode().first()))
    return pl.concat(out).unique(subset="txn_id", keep="first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="zone")
    ap.add_argument("--propagate", action="store_true", help="also settle linked episode rows (rules R1, R2)")
    args = ap.parse_args()
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
    z = pl.read_parquet(CACHE / "device_zone.parquet")
    full = load_raw()
    raw = full.select("txn_id", "label", "split")
    link = full.select("txn_id", "sender_id", "receiver_id", t=pl.col("ts").dt.epoch("s"), is_p2p=pl.col("receiver_type") == "P2P")

    # ---- September check
    fr = [pl.read_parquet(OOF / f"{r}.parquet") for r in args.runs]
    P = np.mean([f.select(LABELS).to_numpy() for f in fr], axis=0)
    o = pl.DataFrame({"txn_id": fr[0]["txn_id"], "pred": P.argmax(1).astype(np.int32)}).join(
        load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left").join(
        raw.select("txn_id", "label"), on="txn_id", how="left", maintain_order="left").join(
        z.filter(pl.col("split") == "train").select("txn_id", "zone"), on="txn_id", how="left", maintain_order="left")
    if args.propagate:
        pr = propagate(o.join(link, on="txn_id", how="left", maintain_order="left").with_columns(
            pred=pl.col("pred").replace_strict(dict(enumerate(LABELS)), return_dtype=pl.String)))
        o = o.join(pr, on="txn_id", how="left", maintain_order="left").with_columns(zone=pl.coalesce("zone", "prop")).drop("prop")
        print(f"propagation (train): {pr.height} off-block rows settled by a linked block row")
    for m, name in ((2, "Sep"), (None, "Jul-Sep")):
        s = o if m is None else o.filter(pl.col("month_fold") == m)
        y = s["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
        p0 = s["pred"].to_numpy()
        zid = s["zone"].replace_strict(LABEL_TO_ID, default=-1, return_dtype=pl.Int32).to_numpy()
        p1 = np.where(zid >= 0, zid, p0)
        # the unreported share: legit-labelled rows on block phones are fraud in clean labels
        y_clean = np.where((zid >= 0) & (y == 0), zid, y)
        f0, f1 = metric.f1_per_class(y, p0), metric.f1_per_class(y, p1)
        g0, g1 = metric.f1_per_class(y_clean, p0), metric.f1_per_class(y_clean, p1)
        print(f"{name}: rows on block phones {int((zid >= 0).sum())}, predictions changed {int((p0 != p1).sum())}")
        print(f"  macro-F1 vs raw labels: {f0.mean():.4f} -> {f1.mean():.4f} ({f1.mean() - f0.mean():+.4f})"
              f"   | counting unreported block rows as fraud: {g0.mean():.4f} -> {g1.mean():.4f} ({g1.mean() - g0.mean():+.4f})")
        if m == 2:
            print(pl.DataFrame({"class": LABELS, "before": f0.round(3), "after": f1.round(3), "diff": (f1 - f0).round(3),
                                "after_unreported_as_fraud": g1.round(3)}))

    # ---- test
    base = pl.read_csv(args.base)
    zt = z.filter(pl.col("split") == "test").select(id="txn_id", zone="zone")
    sub = base.join(zt, on="id", how="left", maintain_order="left")
    if args.propagate:
        pr = propagate(sub.rename({"id": "txn_id", "label": "pred"}).join(link, on="txn_id", how="left", maintain_order="left")).rename({"txn_id": "id"})
        sub = sub.join(pr, on="id", how="left", maintain_order="left").with_columns(zone=pl.coalesce("zone", "prop")).drop("prop")
        print(f"propagation (test): {pr.height} off-block rows settled by a linked block row")
    ch = sub.filter(pl.col("zone").is_not_null() & (pl.col("zone") != pl.col("label")))
    print(f"\ntest: {zt.height} rows on block or bot-pool phones; {ch.height} predictions change")
    print(ch.group_by("label", "zone").agg(n=pl.len()).sort("n", descending=True).rename({"label": "was", "zone": "now"}))
    sub = sub.select("id", label=pl.coalesce("zone", "label"))
    write_submission(sub, f"L2_{args.name}", {"run": f"zone_{args.name}", "base": args.base, "changed": ch.height,
                                              "predicted_fraud_share": float((sub["label"] != "legit").mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
