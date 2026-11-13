"""Group `behaviour_v1`: how a sender and a phone behave (work/L3 tasks 1-3, built on L1).

    python -m src.behaviour_features                 # view half_s0 (default)
    python -m src.behaviour_features --view half_s1

Blocks:
  A  beh_dev_*   bursts on the phone: counts in 1/5/15/60-min windows, regularity (std of the
                 timestamps and of login_to_txn_sec), small amounts, gaming/e-commerce share,
                 distinct accounts on the phone (device farms)                      velocity_bot
  B  beh_snd_*   bursts by the account: counts, distinct phones/cities/receivers, money sent
                 in the next hour against the balance (draining)            sim_swap, ATO, velocity
  C  beh_seq_*   change since the account's previous payment: SIM changed, phone changed, city
                 changed, sim-age drop                                              sim_swap, ATO
  D  beh_sd_*,   has this account used this phone / been in this city before (last 7 days)
     beh_scity_*                                                                    ATO
  E  beh_ref_*   refund claims by the account over 7 and 30 days, refunds per purchase, same
                 amount paid to the same receiver before                            refund_abuse

Every window goes through src/windows.py, so the density rule holds. 'past' is [t - w, t) and
'future' is (t, t + w]: the row itself is never counted there. 'around' windows include the row.
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import polars as pl

from .common import load_raw, save_features
from .windows import attach, build_events, gap_features, window_agg

M = timedelta(minutes=1)
H = timedelta(hours=1)
D = timedelta(days=1)


def _load() -> pl.DataFrame:
    raw = load_raw()
    return raw.select(
        "txn_id", "ts", "month_idx", "split", "sender_id", "receiver_id", "device_id", "device_city",
        "amount", "balance_before", "sim_age_hours", "login_to_txn_sec",
        tsec=pl.col("ts").dt.epoch("s").cast(pl.Float64),
        is_refund=(pl.col("txn_type") == "REFUND").cast(pl.Int32),
        is_purchase=((pl.col("receiver_type") == "MERCHANT") & (pl.col("txn_type") != "REFUND")).cast(pl.Int32),
        is_game_ecom=pl.col("merchant_category").is_in(["gaming", "ecommerce"]).fill_null(False).cast(pl.Int32),
        is_small=(pl.col("amount") < 500).cast(pl.Int32),
        amount_cents=(pl.col("amount") * 100).round(0).cast(pl.Int64),
    )


def _counts(ev, key, prefix, spans):
    out = []
    for name, span in spans.items():
        for d in ("past", "future"):
            out.append(window_agg(ev, key, span, d, [pl.len().cast(pl.Int32).alias(f"{prefix}_n_{d}_{name}")]))
    return out


def block_device(ev7: pl.DataFrame) -> list[pl.DataFrame]:
    k, p = "device_id", "beh_dev"
    frames = _counts(ev7, k, p, {"1m": M, "5m": 5 * M, "15m": 15 * M, "1h": H})
    frames.append(window_agg(ev7, k, 15 * M, "around", [
        pl.col("sender_id").n_unique().alias(f"{p}_senders_15m"),
        pl.col("receiver_id").n_unique().alias(f"{p}_receivers_15m"),
        pl.col("tsec").std().alias(f"{p}_ts_std_15m"),
        pl.col("login_to_txn_sec").std().alias(f"{p}_login_std_15m"),
        pl.col("login_to_txn_sec").mean().alias(f"{p}_login_mean_15m"),
        pl.col("amount").mean().alias(f"{p}_amount_mean_15m"),
        pl.col("amount").std().alias(f"{p}_amount_std_15m"),
        pl.col("is_game_ecom").mean().alias(f"{p}_game_ecom_share_15m"),
        pl.col("is_small").mean().alias(f"{p}_small_share_15m"),
    ]))
    frames.append(window_agg(ev7, k, 7 * D, "around", [
        pl.col("sender_id").n_unique().alias(f"{p}_senders_7d"),
        pl.col("receiver_id").n_unique().alias(f"{p}_receivers_7d"),
        pl.len().cast(pl.Int32).alias(f"{p}_n_7d"),
    ]))
    frames.append(gap_features(ev7, k, p))
    return frames


def block_sender(ev7: pl.DataFrame) -> list[pl.DataFrame]:
    k, p = "sender_id", "beh_snd"
    frames = _counts(ev7, k, p, {"5m": 5 * M, "1h": H, "24h": 24 * H})
    frames.append(window_agg(ev7, k, H, "future", [
        pl.col("amount").sum().alias(f"{p}_amount_future_1h"),
    ]))
    frames.append(window_agg(ev7, k, H, "around", [
        pl.col("device_id").n_unique().alias(f"{p}_devices_1h"),
        pl.col("device_city").n_unique().alias(f"{p}_cities_1h"),
        pl.col("receiver_id").n_unique().alias(f"{p}_receivers_1h"),
        pl.col("tsec").std().alias(f"{p}_ts_std_1h"),
        pl.col("login_to_txn_sec").mean().alias(f"{p}_login_mean_1h"),
    ]))
    frames.append(window_agg(ev7, k, 7 * D, "past", [
        pl.col("device_id").n_unique().alias(f"{p}_devices_past_7d"),
        pl.col("device_city").n_unique().alias(f"{p}_cities_past_7d"),
        pl.col("amount").median().alias(f"{p}_amount_median_past_7d"),
        pl.col("amount").max().alias(f"{p}_amount_max_past_7d"),
    ]))
    frames.append(gap_features(ev7, k, p))
    return frames


def block_sequence(ev7: pl.DataFrame) -> pl.DataFrame:
    """Compare each payment with the same account's previous payment (inside the visible rows)."""
    by = ["vis", "sender_id"]
    s = ev7.sort([*by, "ts", "txn_id"])
    prev = lambda c: pl.col(c).shift(1).over(by)  # noqa: E731
    gap_h = (pl.col("ts") - prev("ts")).dt.total_seconds() / 3600.0
    out = s.with_columns(
        beh_seq_gap_h=gap_h,
        beh_seq_sim_changed=(pl.col("sim_age_hours") < gap_h).cast(pl.Int8),
        beh_seq_sim_drop_h=prev("sim_age_hours") + gap_h - pl.col("sim_age_hours"),
        beh_seq_device_changed=(pl.col("device_id") != prev("device_id")).cast(pl.Int8),
        beh_seq_city_changed=(pl.col("device_city") != prev("device_city")).cast(pl.Int8),
        beh_seq_amount_ratio=pl.col("amount") / prev("amount"),
        beh_seq_balance_drop=prev("balance_before") - pl.col("balance_before"),
    )
    cols = [c for c in out.columns if c.startswith("beh_seq_")]
    return out.filter(pl.col("is_self")).select("txn_id", *cols)


def block_familiarity(ev7: pl.DataFrame) -> list[pl.DataFrame]:
    return [
        window_agg(ev7, ["sender_id", "device_id"], 7 * D, "past",
                   [pl.len().cast(pl.Int32).alias("beh_sd_n_past_7d")]),
        window_agg(ev7, ["sender_id", "device_city"], 7 * D, "past",
                   [pl.len().cast(pl.Int32).alias("beh_scity_n_past_7d")]),
        window_agg(ev7, "sender_id", 7 * D, "past", [pl.len().cast(pl.Int32).alias("beh_snd_n_past_7d")]),
    ]


def block_refund(ev31: pl.DataFrame) -> list[pl.DataFrame]:
    k, p = "sender_id", "beh_ref"
    frames = []
    for name, span in {"7d": 7 * D, "30d": 30 * D}.items():
        for d in ("past", "future"):
            frames.append(window_agg(ev31, k, span, d, [
                pl.col("is_refund").sum().cast(pl.Int32).alias(f"{p}_n_{d}_{name}"),
                pl.col("is_purchase").sum().cast(pl.Int32).alias(f"{p}_purchases_{d}_{name}"),
            ]))
    frames.append(window_agg(ev31, k, 30 * D, "around", [
        pl.col("is_game_ecom").mean().alias(f"{p}_game_ecom_share_30d"),
        pl.col("receiver_id").n_unique().alias(f"{p}_receivers_30d"),
    ]))
    frames.append(window_agg(ev31, ["sender_id", "receiver_id", "amount_cents"], 30 * D, "past",
                             [pl.len().cast(pl.Int32).alias(f"{p}_same_amount_pair_past_30d")]))
    frames.append(window_agg(ev31, ["sender_id", "receiver_id"], 30 * D, "past", [
        pl.len().cast(pl.Int32).alias(f"{p}_pair_n_past_30d"),
        pl.col("is_refund").sum().cast(pl.Int32).alias(f"{p}_pair_refunds_past_30d"),
    ]))
    return frames


def build(view: str) -> pl.DataFrame:
    t0 = time.time()
    raw = _load()
    ev7 = build_events(raw, view, max_window=7 * D)
    frames = block_device(ev7)
    print(f"  device block {time.time() - t0:.0f}s")
    frames += block_sender(ev7)
    frames.append(block_sequence(ev7))
    frames += block_familiarity(ev7)
    print(f"  sender blocks {time.time() - t0:.0f}s")
    del ev7
    ev31 = build_events(raw, view, max_window=31 * D)
    frames += block_refund(ev31)
    print(f"  refund block {time.time() - t0:.0f}s")
    out = attach(raw, *frames)

    # ratios computed after the join (all inputs are in the same view)
    r = raw.select("txn_id", "amount", "balance_before")
    out = out.join(r, on="txn_id", how="left", maintain_order="left").with_columns(
        beh_snd_drain_future_1h=(pl.col("beh_snd_amount_future_1h") + pl.col("amount")) / (pl.col("balance_before") + 1.0),
        beh_snd_amount_vs_median_7d=pl.col("amount") / pl.col("beh_snd_amount_median_past_7d"),
        beh_sd_new_device=((pl.col("beh_sd_n_past_7d").fill_null(0) == 0)
                           & (pl.col("beh_snd_n_past_7d").fill_null(0) > 0)).cast(pl.Int8),
        beh_scity_new_city=((pl.col("beh_scity_n_past_7d").fill_null(0) == 0)
                            & (pl.col("beh_snd_n_past_7d").fill_null(0) > 0)).cast(pl.Int8),
        beh_ref_per_purchase_30d=(pl.col("beh_ref_n_past_30d") + pl.col("beh_ref_n_future_30d"))
        / (pl.col("beh_ref_purchases_past_30d") + pl.col("beh_ref_purchases_future_30d") + 1.0),
    ).drop("amount", "balance_before")
    print(f"  done {time.time() - t0:.0f}s, {out.width - 1} features")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="half_s0")
    a = ap.parse_args()
    save_features(build(a.view), "behaviour_v1", a.view)
