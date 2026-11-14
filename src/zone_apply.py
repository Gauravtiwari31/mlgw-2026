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
from .common import CACHE, LABEL_TO_ID, LABELS, N_CLASSES, OOF, ROOT, load_folds, load_raw
from .postprocess import write_submission


FANIN = ["mule_account", "velocity_bot", "layering_chain"]
TAKEOVER = ["sim_swap", "account_takeover"]
COLLECT = ["phishing_collect", "fake_cashback"]


def propagate(rows: pl.DataFrame, use: tuple[str, ...] = ("R1", "R2", "R3", "R4")) -> pl.DataFrame:
    """rows: txn_id, t (epoch s), sender_id, receiver_id, is_p2p, pred (class name), zone (null off the block).
    Returns txn_id, prop for off-block rows whose class is settled by a block row of the same episode.
    A rule fires only when the row's prediction and the block row's zone are in the same look-alike group:
      R1  same personal receiver within 24 h                          mule / velocity / layering
      R2  same sender within 2 h                                      sim_swap / account_takeover
      R3  the reverse payment of the pair within 24 h                 phishing / cashback
      R4  the row pays the block row's sender within 2 h, or is paid
          on by the block row's receiver within 24 h (money moving on) mule / velocity / layering
    Train check (audit/device_blocks.md): where rule and blend disagree the rule is right in 98% (R1 to
    velocity), 100% (R2, R3) and 13 of 14 (R4). Wider windows and other links are worse than the blend."""
    blk = rows.filter(pl.col("zone").is_not_null())
    rest = rows.filter(pl.col("zone").is_null())
    p2p = pl.col("is_p2p")
    rules = [  # (rule, row keys, block keys, window s, group, zones allowed, P2P only)
        ("R1", ["receiver_id"], ["receiver_id"], 86400, FANIN, FANIN, True),
        ("R2", ["sender_id"], ["sender_id"], 7200, TAKEOVER, TAKEOVER, False),
        ("R3", ["sender_id", "receiver_id"], ["receiver_id", "sender_id"], 86400, COLLECT, COLLECT, True),
        ("R4", ["receiver_id"], ["sender_id"], 7200, FANIN, ["mule_account", "layering_chain"], True),
        ("R4", ["sender_id"], ["receiver_id"], 86400, FANIN, FANIN, True),
    ]
    out = []
    for rid, rk, bk, win, group, zones, only_p2p in rules:
        if rid not in use:
            continue
        r = rest.filter(pl.col("pred").is_in(group) & (p2p if only_p2p else pl.lit(True)))
        b = blk.filter(pl.col("zone").is_in(zones) & (p2p if only_p2p else pl.lit(True)))
        r = r.select("txn_id", "t", *[pl.col(c).alias(f"k{i}") for i, c in enumerate(rk)])
        b = b.select(*[pl.col(c).alias(f"k{i}") for i, c in enumerate(bk)], bt="t", bz="zone")
        j = r.join(b, on=[f"k{i}" for i in range(len(rk))], how="inner").filter((pl.col("t") - pl.col("bt")).abs() <= win)
        out.append(j.group_by("txn_id").agg(prop=pl.col("bz").mode().first()))
    return pl.concat(out).unique(subset="txn_id", keep="first", maintain_order=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="zone")
    ap.add_argument("--propagate", action="store_true", help="also settle linked episode rows (rules R1 to R4)")
    ap.add_argument("--rules", default="R1,R2,R3,R4", help="which link rules to use with --propagate (the 0.5403 file used R1,R2)")
    args = ap.parse_args()
    use = tuple(args.rules.split(","))
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
            pred=pl.col("pred").replace_strict(dict(enumerate(LABELS)), return_dtype=pl.String)), use)
        o = o.join(pr, on="txn_id", how="left", maintain_order="left").with_columns(zone=pl.coalesce("zone", "prop")).drop("prop")
        print(f"propagation (train): {pr.height} off-block rows settled by a linked block row")
    # test labels are clean (PROBLEM_STATEMENT): no unreported rows. Test-like ruler = raw labels, with the rows
    # labels_clean flags as unreported counted as fraud (type: zone if on the block, else the blend's own type).
    st = o.select("txn_id").join(pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "status"), on="txn_id", how="left",
                                 maintain_order="left")["status"].is_in(["unreported", "unreported_typed"]).fill_null(False).to_numpy()
    blend_type = 1 + P[:, 1:].argmax(1)
    o = o.with_columns(unrep=pl.Series(st), btype=pl.Series(blend_type.astype(np.int32)))
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
        y_like = np.where(s["unrep"].to_numpy() & (y == 0), np.where(zid >= 0, zid, s["btype"].to_numpy()), y)
        print(f"{name}: rows on block phones or linked {int((zid >= 0).sum())}, predictions changed {int((p0 != p1).sum())}")
        print(f"  macro-F1 test-like (unreported rows counted as fraud): {metric.macro_f1(y_like, p0):.4f} -> {metric.macro_f1(y_like, p1):.4f}")
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
        pr = propagate(sub.rename({"id": "txn_id", "label": "pred"}).join(link, on="txn_id", how="left", maintain_order="left"), use).rename({"txn_id": "id"})
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
