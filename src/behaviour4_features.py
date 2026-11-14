"""Group `behaviour_v4` (L3): what CHANGED, relative to the sender's own 30-day baseline.

The team's error analysis (audit/leaderboard_analysis.md): sim_swap and account_takeover are each
other's main confusion (about 30% of their train labels carry the other type's signature) and are
the weakest classes. The two fraud types differ by WHAT changed:
  - sim_swap: the SAME phone, a NEW SIM (sim_age fresh while the device is already known to the
    account), night drain.
  - account_takeover: a NEW phone never seen for this account, far from home.
Every feature below is sender-relative (how unusual for ME) and pair-relative (this device for me),
computed through src/windows.py so the density rule holds.

Features (prefix beh4_):
  beh4_pair_n_30d             payments on this (sender, device) pair in the past 30 d (0 = device new to me)
  beh4_pair_secs_since_prev    time since the previous payment on this pair
  beh4_prev_dev_same           sender's previous payment (any device) was on this device
  beh4_sim_med_30d             sender's median sim_age_hours, past 30 d
  beh4_sim_age_vs_med          this row's sim_age_hours minus that median (negative = fresher than usual)
  beh4_sim_newer_than_dev      sim_age_hours < device_age_days * 24 (SIM swapped while phone already paired)
  beh4_dist_med_30d            sender's median distance-from-home, past 30 d
  beh4_dist_vs_med             this row's distance minus that median (positive = suddenly far: ATO)
  beh4_amt_med_30d             sender's median amount, past 30 d
  beh4_amt_vs_med              amount / median (drain pattern)
  beh4_bal_med_30d             sender's median balance_before, past 30 d
  beh4_bal_vs_med              balance_before / median (near-empty account: drain)
  beh4_pin_med_30d             sender's median pin_retries, past 30 d
  beh4_pin_vs_med              pin_retries minus median (auth struggle on the new SIM/device)
  beh4_night_share_30d         sender's night-payment share, past 30 d
  beh4_is_night_vs_share       is_night minus that share (drain at unusual hours for ME)
"""
from __future__ import annotations

import math
from datetime import timedelta

import polars as pl

from .common import load_accounts, load_raw, save_features
from .windows import build_events, gap_features, window_agg

D = timedelta(days=1)
R_EARTH = 6371.0
RAD = math.pi / 180.0

CITY_COORDS = {
    "Agra": (27.1767, 78.0081), "Ahmedabad": (23.0225, 72.5714), "Alwar": (27.5530, 76.6347),
    "Amritsar": (31.6340, 74.8723), "Bengaluru": (12.9716, 77.5946), "Bharatpur": (27.2152, 77.5030),
    "Bhopal": (23.2599, 77.4126), "Chandigarh": (30.7333, 76.7794), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Dehradun": (30.3165, 78.0322), "Delhi": (28.6139, 77.2090),
    "Deoghar": (24.4744, 86.7000), "Giridih": (24.1853, 86.3000), "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867), "Indore": (22.7196, 75.8577), "Jaipur": (26.9124, 75.7873),
    "Jamtara": (23.9437, 86.6262), "Jodhpur": (26.2389, 73.0243), "Kanpur": (26.4499, 80.3319),
    "Kochi": (9.9312, 76.2673), "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462),
    "Ludhiana": (30.9010, 75.8573), "Madurai": (9.9252, 78.1198), "Mathura": (27.4924, 77.6737),
    "Mewat": (28.1000, 76.9900), "Mumbai": (19.0760, 72.8777), "Mysuru": (12.2958, 76.6394),
    "Nagpur": (21.1458, 79.0882), "Nashik": (19.9975, 73.7898), "Patna": (25.5941, 85.1376),
    "Pune": (18.5204, 73.8567), "Raipur": (21.2514, 81.6296), "Ranchi": (23.3441, 85.3096),
    "Surat": (21.1702, 72.8311), "Vadodara": (22.3072, 73.1812), "Varanasi": (25.3176, 82.9739),
    "Visakhapatnam": (17.6868, 83.2185),
}


def _dist_km(lat_a, lon_a, lat_b, lon_b):
    p1, p2 = pl.col(lat_a) * RAD, pl.col(lat_b) * RAD
    dp, dl = p2 - p1, (pl.col(lon_b) - pl.col(lon_a)) * RAD
    a = (dp / 2).sin().pow(2) + p1.cos() * p2.cos() * (dl / 2).sin().pow(2)
    return 2 * R_EARTH * a.clip(0.0, 1.0).sqrt().arcsin()


def build(view: str) -> pl.DataFrame:
    raw = load_raw()
    acc = load_accounts()
    coords = pl.DataFrame({
        "city": list(CITY_COORDS), "lat": [v[0] for v in CITY_COORDS.values()],
        "lon": [v[1] for v in CITY_COORDS.values()],
    })
    base = (
        raw.select("txn_id", "ts", "month_idx", "sender_id", "device_id", "device_city",
                   "sim_age_hours", "device_age_days", "pin_retries", "amount", "balance_before")
        .with_columns(
            hour=pl.col("ts").dt.hour(),
            is_night=(pl.col("ts").dt.hour() < 6).cast(pl.Int8),
        )
        .join(acc.select(sender_id="account_id", s_home="home_city"), on="sender_id",
              how="left", maintain_order="left")
        .join(coords.rename({"city": "device_city", "lat": "d_lat", "lon": "d_lon"}),
              on="device_city", how="left", maintain_order="left")
        .join(coords.rename({"city": "s_home", "lat": "s_lat", "lon": "s_lon"}),
              on="s_home", how="left", maintain_order="left")
        .with_columns(dist_home=_dist_km("d_lat", "d_lon", "s_lat", "s_lon"))
    )

    ev = build_events(base, view, max_window=31 * D)
    f_snd = window_agg(
        ev, "sender_id", 30 * D, "past",
        [pl.col("device_id").last().alias("beh4_prev_dev"),
         pl.col("sim_age_hours").median().alias("beh4_sim_med_30d"),
         pl.col("dist_home").median().alias("beh4_dist_med_30d"),
         pl.col("amount").median().alias("beh4_amt_med_30d"),
         pl.col("balance_before").median().alias("beh4_bal_med_30d"),
         pl.col("pin_retries").median().alias("beh4_pin_med_30d"),
         pl.col("is_night").mean().alias("beh4_night_share_30d"),
         pl.col("device_id").n_unique().alias("beh4_snd_devs_30d")],
    )
    f_pair = window_agg(
        ev, ["sender_id", "device_id"], 30 * D, "past",
        [pl.len().alias("beh4_pair_n_30d")],
    )
    gap_pair = gap_features(ev, ["sender_id", "device_id"], "beh4_pair")

    out = (
        base
        .join(f_snd, on="txn_id", how="left", maintain_order="left")
        .join(f_pair, on="txn_id", how="left", maintain_order="left")
        .join(gap_pair, on="txn_id", how="left", maintain_order="left")
    )
    out = out.with_columns(
        beh4_prev_dev_same=(pl.col("beh4_prev_dev") == pl.col("device_id")).cast(pl.Int8),
        beh4_sim_age_vs_med=pl.col("sim_age_hours") - pl.col("beh4_sim_med_30d"),
        beh4_sim_newer_than_dev=(pl.col("sim_age_hours")
                                 < pl.col("device_age_days") * 24.0).cast(pl.Int8),
        beh4_dist_vs_med=pl.col("dist_home") - pl.col("beh4_dist_med_30d"),
        beh4_amt_vs_med=pl.col("amount") / (pl.col("beh4_amt_med_30d") + 1.0),
        beh4_bal_vs_med=pl.col("balance_before") / (pl.col("beh4_bal_med_30d") + 1.0),
        beh4_pin_vs_med=pl.col("pin_retries") - pl.col("beh4_pin_med_30d"),
        beh4_is_night_vs_share=pl.col("is_night") - pl.col("beh4_night_share_30d"),
        beh4_pair_n_30d=pl.col("beh4_pair_n_30d").fill_null(0),
    )

    return out.select(
        "txn_id", "beh4_pair_n_30d", "beh4_pair_secs_since_prev", "beh4_prev_dev_same",
        "beh4_sim_med_30d", "beh4_sim_age_vs_med", "beh4_sim_newer_than_dev",
        "beh4_dist_med_30d", "beh4_dist_vs_med", "beh4_amt_med_30d", "beh4_amt_vs_med",
        "beh4_bal_med_30d", "beh4_bal_vs_med", "beh4_pin_med_30d", "beh4_pin_vs_med",
        "beh4_night_share_30d", "beh4_is_night_vs_share", "beh4_snd_devs_30d",
    )


if __name__ == "__main__":
    import argparse

    from .common import VIEWS

    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="fw_s0", choices=VIEWS)
    a = ap.parse_args()
    save_features(build(a.view), "behaviour_v4", a.view)
