"""Group `stage2_v1`: neighbours' stage-1 predictions (research.md D4 M5, work/L4 task 9).

For each row: the average stage-1 class probabilities of the OTHER rows that share its sender, device,
receiver (person-to-person only) or sender-receiver pair within +-24 hours, in the chosen view.
Train rows use out-of-fold probabilities (each month predicted by a model that never saw that month's
labels); test rows use the test predictions. So no row's own label, or its month's labels, leak in.

    python -m src.stage2_features --runs RUN ... --view fw_s0
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import numpy as np
import polars as pl

from .common import LABELS, OOF, PREDS, load_raw, save_features
from .windows import attach, build_events, window_agg

SHORT = {"legit": "legit", "mule_account": "mule", "velocity_bot": "velo", "layering_chain": "layer", "sim_swap": "sim",
         "account_takeover": "ato", "phishing_collect": "phish", "qr_swap": "qr", "refund_abuse": "refund",
         "fake_cashback": "cash", "loan_app_extortion": "loan", "merchant_collusion": "collude"}


def stage1(runs: list[str]) -> pl.DataFrame:
    def avg(d):
        fr = [pl.read_parquet(d / f"{r}.parquet") for r in runs]
        P = np.mean([f.select(LABELS).to_numpy() for f in fr], axis=0)
        return pl.DataFrame({"txn_id": fr[0]["txn_id"]}).with_columns([pl.Series(f"p_{SHORT[c]}", P[:, i].astype(np.float32)) for i, c in enumerate(LABELS)])
    return pl.concat([avg(OOF), avg(PREDS)])


def build(runs: list[str], view: str) -> pl.DataFrame:
    t0 = time.time()
    raw = load_raw().select("txn_id", "ts", "month_idx", "sender_id", "receiver_id", "device_id", "receiver_type")
    p = stage1(runs)
    df = raw.join(p, on="txn_id", how="left", maintain_order="left")
    pc = [c for c in df.columns if c.startswith("p_")]
    df = df.with_columns(pl.col(pc).fill_null(0.0), pair=pl.concat_str(["sender_id", "receiver_id"], separator="|"),
                         rcv_p2p=pl.when(pl.col("receiver_type") == "P2P").then(pl.col("receiver_id")))
    ev = build_events(df, view, max_window=timedelta(days=1))
    D = timedelta(hours=24)
    frames = []
    for key, tag in (("sender_id", "snd"), ("device_id", "dev"), ("rcv_p2p", "rcv"), ("pair", "pair")):
        aggs = [pl.len().alias(f"s2_{tag}_n")] + [pl.col(c).sum().alias(f"s2_{tag}_sum_{c[2:]}") for c in pc]
        f = window_agg(ev, key, D, "around", aggs)
        frames.append(f)
    out = attach(raw, *frames)
    out = out.join(df.select("txn_id", *pc), on="txn_id", how="left", maintain_order="left")
    exprs = []
    for tag in ("snd", "dev", "rcv", "pair"):
        n = pl.col(f"s2_{tag}_n") - 1                                  # other rows (the window includes the row itself)
        exprs.append(n.alias(f"s2_{tag}_others"))
        for c in pc:
            k = c[2:]
            exprs.append(pl.when(n > 0).then((pl.col(f"s2_{tag}_sum_{k}") - pl.col(c)) / n).alias(f"s2_{tag}_{k}"))
    out = out.select("txn_id", *exprs)
    print(f"  stage2 features: {out.width - 1} in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--view", default="fw_s0")
    a = ap.parse_args()
    save_features(build(a.runs, a.view), "stage2_v1", a.view)
