"""Group `behaviour_v3` (L3): city distance and travel-speed features.

Motivation (work/L3 TASKS.md #6 + STATUS 13:40):
- account_takeover: the fraudster pays from a phone in a city far from the victim's home, often
  hours after the victim's own last payment somewhere else -> distance and impossible-travel speed.
- sim_swap: same shape, plus fresh SIM (already in base); distance adds magnitude over the existing
  away_from_home flag.
- The city list includes known fraud-belt towns (Jamtara, Mewat, Bharatpur, Giridih, Deoghar,
  Alwar), so raw distance from home carries signal even for non-ATO classes.

All relational aggregates go through src/windows.py (density rule, one row per txn_id).
City coordinates are public reference data written in code (allowed, research.md T9).
Features (prefix beh3_):
  beh3_dist_device_home_km      haversine km, device city vs sender home city
  beh3_dist_device_home_log     log1p of the above
  beh3_dist_device_rcv_km      device city vs receiver home city (P2P) or merchant city
  beh3_snd_prev_city_km        device city vs the city of the sender's previous payment (30 d)
  beh3_snd_speed_kmh           that distance / hours since the sender's previous payment (capped)
  beh3_snd_speed_log           log1p of the above
  beh3_snd_dist_home_mean_30d  mean dist-from-home of the sender's payments in the past 30 d
  beh3_dist_home_delta         this row's dist-from-home minus that mean (how unusual for me)
  beh3_snd_cities_30d          distinct device cities of this sender in the past 30 d
  beh3_dev_dist_home_mean_30d  mean dist-from-home on this device's past 30 d (phones used by far-apart people)
  beh3_dev_dist_home_max_30d   max dist-from-home on this device's past 30 d
  beh3_dev_home_cities_30d     distinct sender home cities on this device in the past 30 d
"""
from __future__ import annotations

import math
from datetime import timedelta

import polars as pl

from .common import load_accounts, load_merchants, load_raw, save_features
from .windows import build_events, gap_features, window_agg

D = timedelta(days=1)

# Approximate city-centre coordinates, public reference data (research.md T9 allows this table).
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
R_EARTH = 6371.0
RAD = math.pi / 180.0


def _coords_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"city": list(CITY_COORDS), "lat": [v[0] for v in CITY_COORDS.values()],
         "lon": [v[1] for v in CITY_COORDS.values()]}
    )


def _dist_km(lat_a: str, lon_a: str, lat_b: str, lon_b: str) -> pl.Expr:
    """Haversine distance in km between two (lat, lon) column pairs, null-safe."""
    p1, p2 = pl.col(lat_a) * RAD, pl.col(lat_b) * RAD
    dp, dl = p2 - p1, (pl.col(lon_b) - pl.col(lon_a)) * RAD
    a = (dp / 2).sin().pow(2) + p1.cos() * p2.cos() * (dl / 2).sin().pow(2)
    return (2 * R_EARTH * a.clip(0.0, 1.0).sqrt().arcsin()).alias("_dist")


def build(view: str) -> pl.DataFrame:
    raw = load_raw()
    acc = load_accounts()
    mer = load_merchants()
    coords = _coords_frame()

    base = (
        raw.select("txn_id", "ts", "month_idx", "sender_id", "device_id", "receiver_id",
                   "receiver_type", "device_city")
        .join(acc.select(sender_id="account_id", s_home="home_city"), on="sender_id",
              how="left", maintain_order="left")
        .join(acc.select(receiver_id="account_id", r_home="home_city"), on="receiver_id",
              how="left", maintain_order="left")
        .join(mer.select(receiver_id="merchant_id", m_city="city"), on="receiver_id",
              how="left", maintain_order="left")
        .with_columns(rcv_city=pl.coalesce("r_home", "m_city"))
        # coordinates for the three cities of this row
        .join(coords.rename({"city": "device_city", "lat": "d_lat", "lon": "d_lon"}),
              on="device_city", how="left", maintain_order="left")
        .join(coords.rename({"city": "s_home", "lat": "s_lat", "lon": "s_lon"}),
              on="s_home", how="left", maintain_order="left")
        .join(coords.rename({"city": "rcv_city", "lat": "r_lat", "lon": "r_lon"}),
              on="rcv_city", how="left", maintain_order="left")
    )
    # per-row distances (computed before build_events so the window means work)
    base = base.with_columns(
        _dist_km("d_lat", "d_lon", "s_lat", "s_lon").alias("dist_home"),
        _dist_km("d_lat", "d_lon", "r_lat", "r_lon").alias("dist_rcv"),
    )

    # relational features: past 30 days per sender and per device (density rule via build_events)
    ev = build_events(base, view, max_window=31 * D)
    f_snd = window_agg(
        ev, "sender_id", 30 * D, "past",
        [pl.col("device_city").last().alias("beh3_snd_prev_city"),
         pl.col("device_city").n_unique().alias("beh3_snd_cities_30d"),
         pl.col("dist_home").mean().alias("beh3_snd_dist_home_mean_30d")],
    )
    f_dev = window_agg(
        ev, "device_id", 30 * D, "past",
        [pl.col("dist_home").mean().alias("beh3_dev_dist_home_mean_30d"),
         pl.col("dist_home").max().alias("beh3_dev_dist_home_max_30d"),
         pl.col("s_home").n_unique().alias("beh3_dev_home_cities_30d")],
    )
    gap = gap_features(ev, "sender_id", "beh3_snd")

    out = (
        base
        .join(f_snd, on="txn_id", how="left", maintain_order="left")
        .join(f_dev, on="txn_id", how="left", maintain_order="left")
        .join(gap, on="txn_id", how="left", maintain_order="left")
    )
    out = out.join(coords.rename({"city": "beh3_snd_prev_city", "lat": "p_lat", "lon": "p_lon"}),
                   on="beh3_snd_prev_city", how="left", maintain_order="left")
    out = out.with_columns(
        beh3_dist_device_home_km=pl.col("dist_home"),
        beh3_dist_device_home_log=pl.col("dist_home").log1p(),
        beh3_dist_device_rcv_km=pl.col("dist_rcv"),
        beh3_snd_prev_city_km=_dist_km("d_lat", "d_lon", "p_lat", "p_lon"),
        beh3_dist_home_delta=(pl.col("dist_home") - pl.col("beh3_snd_dist_home_mean_30d")),
    )
    out = out.with_columns(
        beh3_snd_speed_kmh=(pl.col("beh3_snd_prev_city_km")
                            / (pl.col("beh3_snd_secs_since_prev") / 3600.0))
                            .clip(0.0, 5000.0)
                            .fill_nan(0.0),  # 0 km / 0 h: same city, same second -> no travel
    )
    out = out.with_columns(beh3_snd_speed_log=pl.col("beh3_snd_speed_kmh").log1p())

    return out.select(
        "txn_id", "beh3_dist_device_home_km", "beh3_dist_device_home_log", "beh3_dist_device_rcv_km",
        "beh3_snd_prev_city_km", "beh3_snd_speed_kmh", "beh3_snd_speed_log",
        "beh3_snd_dist_home_mean_30d", "beh3_dist_home_delta", "beh3_snd_cities_30d",
        "beh3_dev_dist_home_mean_30d", "beh3_dev_dist_home_max_30d", "beh3_dev_home_cities_30d",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="half_s0", choices=("half_s0", "half_s1", "full"))
    a = ap.parse_args()
    save_features(build(a.view), "behaviour_v3", a.view)
