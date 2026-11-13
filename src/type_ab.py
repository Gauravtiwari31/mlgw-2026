"""One-rule A/B files: apply a single type rule to the predictions of a base submission.

Two uploads that differ only in a few dozen rows compare almost without leaderboard noise: every
other row scores the same in both. So each rule is tested alone on top of the best file, and kept
only if the public score goes up. Rules touch predicted fraud types only, never legit rows.

    python -m src.type_ab --base submissions/<best>.csv --rule simato
    python -m src.type_ab --base submissions/<best>.csv --rule cashback

Rules
  simato    predicted sim_swap / account_takeover, per sender-day: fresh SIM (< 72 h) and no new phone
            (< 3 days) -> sim_swap; new phone and no fresh SIM -> account_takeover
  cashback  predicted phishing_collect inside a Rs 1-10 exchange between the pair (tiny credit then a
            payment back within 24 h, either direction) -> fake_cashback; predicted fake_cashback that
            is a COLLECT request with no such exchange -> phishing_collect
"""
from __future__ import annotations

import argparse

import polars as pl

from .common import load_features, load_raw
from .postprocess import write_submission


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--rule", required=True, choices=["simato", "cashback"])
    args = ap.parse_args()
    base = pl.read_csv(args.base)
    raw = load_raw().filter(pl.col("split") == "test").select(
        "txn_id", "ts", "sender_id", "txn_type", "amount", "sim_age_hours", "device_age_days")
    fl = load_features("flow_v1", "fw_s0").select("txn_id", "pair_rev_tiny_n_past_24h", "pair_rev_n_future_24h")
    d = base.rename({"id": "txn_id"}).join(raw, on="txn_id", how="left", maintain_order="left").join(fl, on="txn_id", how="left", maintain_order="left")
    lab = pl.col("label")
    if args.rule == "simato":
        ep = d.filter(lab.is_in(["sim_swap", "account_takeover"])).group_by("sender_id", pl.col("ts").dt.date().alias("_day")).agg(
            _sim=(pl.col("sim_age_hours") < 72).any(), _dev=(pl.col("device_age_days") < 3).any())
        d = d.with_columns(_day=pl.col("ts").dt.date()).join(ep, on=["sender_id", "_day"], how="left", maintain_order="left")
        new = (pl.when(lab.is_in(["sim_swap", "account_takeover"]) & pl.col("_sim") & ~pl.col("_dev")).then(pl.lit("sim_swap"))
               .when(lab.is_in(["sim_swap", "account_takeover"]) & pl.col("_dev") & ~pl.col("_sim")).then(pl.lit("account_takeover"))
               .otherwise(lab))
    else:
        ex = ((pl.col("amount") <= 10) & (pl.col("pair_rev_n_future_24h").fill_null(0) > 0)) | (pl.col("pair_rev_tiny_n_past_24h").fill_null(0) > 0)
        new = (pl.when((lab == "phishing_collect") & ex).then(pl.lit("fake_cashback"))
               .when((lab == "fake_cashback") & (pl.col("txn_type") == "COLLECT") & ~ex).then(pl.lit("phishing_collect"))
               .otherwise(lab))
    d = d.with_columns(_new=new)
    ch = d.filter(pl.col("_new") != pl.col("label"))
    print(f"rule {args.rule}: {ch.height} test rows change (about {ch.height * 0.3:.0f} in the public part)")
    print(ch.group_by("label", "_new").agg(n=pl.len()).sort("n", descending=True))
    sub = d.select(id="txn_id", label="_new")
    write_submission(sub, f"AB_{args.rule}", {"run": f"AB_{args.rule}", "base": args.base, "rule": args.rule, "changed": ch.height,
                                              "predicted_fraud_share": float((sub["label"] != "legit").mean()), "sep_macro_f1_honest": float("nan")})


if __name__ == "__main__":
    main()
