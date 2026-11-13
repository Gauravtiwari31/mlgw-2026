"""Group `behaviour_v2`: repeated payments and collect requests (work/L3 tasks 4-5, built on L1).

    python -m src.behaviour2_features                 # view half_s0 (default)
    python -m src.behaviour2_features --view half_s1

Blocks:
  A  b2_pair_*   the same sender paying the same receiver again and again: counts over 1/7/30
                 days both sides, distinct days, evening share, amount spread       loan_app_extortion
  B  b2_rcv_*    the receiver's side: how many senders pay it repeatedly, evening share, lending
                 share, round-number share                       loan_app_extortion, merchant_collusion
  C  b2_col_*    COLLECT requests: collect fan-in at the receiver (distinct victims), share of
                 collects just under a round limit, collects the sender approved     phishing_collect

Same window rules as every group: src/windows.py, density rule, 'past'/'future' exclude the row.
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import polars as pl

from .common import load_raw, save_features
from .windows import attach, build_events, window_agg

H = timedelta(hours=1)
D = timedelta(days=1)


def _load() -> pl.DataFrame:
    raw = load_raw()
    amt = pl.col("amount")
    near_round = pl.any_horizontal([((amt / k).ceil() * k - amt).is_between(0.01, k * 0.03) for k in (500, 1000, 2000, 5000, 10000)])
    hour = pl.col("ts").dt.hour()
    return raw.select(
        "txn_id", "ts", "month_idx", "split", "sender_id", "receiver_id", "amount",
        day=pl.col("ts").dt.date(),
        is_evening=(hour >= 20).cast(pl.Int32),
        is_night=(hour < 6).cast(pl.Int32),
        is_lending=(pl.col("merchant_category") == "lending").fill_null(False).cast(pl.Int32),
        is_collect=(pl.col("txn_type") == "COLLECT").cast(pl.Int32),
        is_round_100=((amt * 100).round(0).cast(pl.Int64) % 10000 == 0).cast(pl.Int32),
        near_round=near_round.cast(pl.Int32),
    )


def block_pair(ev: pl.DataFrame) -> list[pl.DataFrame]:
    k, p = ["sender_id", "receiver_id"], "b2_pair"
    frames = []
    for name, span in {"1d": D, "7d": 7 * D, "30d": 30 * D}.items():
        for d in ("past", "future"):
            frames.append(window_agg(ev, k, span, d, [pl.len().cast(pl.Int32).alias(f"{p}_n_{d}_{name}")]))
    frames.append(window_agg(ev, k, 30 * D, "around", [
        pl.col("day").n_unique().alias(f"{p}_days_30d"),
        pl.col("is_evening").mean().alias(f"{p}_evening_share_30d"),
        pl.col("amount").std().alias(f"{p}_amount_std_30d"),
        pl.col("amount").mean().alias(f"{p}_amount_mean_30d"),
    ]))
    return frames


def block_receiver(ev: pl.DataFrame) -> list[pl.DataFrame]:
    k, p = "receiver_id", "b2_rcv"
    return [
        window_agg(ev, k, 30 * D, "around", [
            pl.len().cast(pl.Int32).alias(f"{p}_n_30d"),
            pl.col("sender_id").n_unique().alias(f"{p}_senders_30d"),
            pl.col("is_evening").mean().alias(f"{p}_evening_share_30d"),
            pl.col("is_night").mean().alias(f"{p}_night_share_30d"),
            pl.col("is_lending").mean().alias(f"{p}_lending_share_30d"),
            pl.col("is_round_100").mean().alias(f"{p}_round100_share_30d"),
            pl.col("amount").std().alias(f"{p}_amount_std_30d"),
        ]),
        window_agg(ev, k, 7 * D, "around", [
            pl.len().cast(pl.Int32).alias(f"{p}_n_7d"),
            pl.col("sender_id").n_unique().alias(f"{p}_senders_7d"),
            pl.col("is_round_100").mean().alias(f"{p}_round100_share_7d"),
        ]),
    ]


def block_collect(ev: pl.DataFrame) -> list[pl.DataFrame]:
    is_c = pl.col("is_collect") == 1
    frames = []
    for name, span in {"1h": H, "24h": 24 * H, "7d": 7 * D}.items():
        frames.append(window_agg(ev, "receiver_id", span, "around", [
            pl.col("is_collect").sum().cast(pl.Int32).alias(f"b2_col_rcv_n_{name}"),
            pl.col("sender_id").filter(is_c).n_unique().alias(f"b2_col_rcv_senders_{name}"),
            pl.col("near_round").filter(is_c).mean().alias(f"b2_col_rcv_near_round_share_{name}"),
        ]))
    frames.append(window_agg(ev, "sender_id", 7 * D, "around", [
        pl.col("is_collect").sum().cast(pl.Int32).alias("b2_col_snd_n_7d"),
        pl.col("receiver_id").filter(is_c).n_unique().alias("b2_col_snd_receivers_7d"),
    ]))
    return frames


def build(view: str) -> pl.DataFrame:
    t0 = time.time()
    raw = _load()
    ev = build_events(raw, view, max_window=31 * D)
    frames = block_pair(ev)
    print(f"  pair block {time.time() - t0:.0f}s")
    frames += block_receiver(ev)
    frames += block_collect(ev)
    print(f"  receiver + collect blocks {time.time() - t0:.0f}s")
    out = attach(raw, *frames).with_columns(
        b2_pair_repeat_share_30d=(pl.col("b2_pair_n_past_30d") + pl.col("b2_pair_n_future_30d"))
        / (pl.col("b2_rcv_n_30d") + 1.0),
        b2_rcv_payments_per_sender_30d=pl.col("b2_rcv_n_30d") / pl.col("b2_rcv_senders_30d"),
        b2_rcv_payments_per_sender_7d=pl.col("b2_rcv_n_7d") / pl.col("b2_rcv_senders_7d"),
    )
    print(f"  done {time.time() - t0:.0f}s, {out.width - 1} features")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="half_s0")
    a = ap.parse_args()
    save_features(build(a.view), "behaviour_v2", a.view)
