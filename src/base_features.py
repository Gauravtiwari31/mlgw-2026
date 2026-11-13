"""Group `base_v1`: single-row features. One view only ('full'), since no other rows are used.

Rules applied (research.md A2-A4):
- raw app_version is dropped
- ip_risk_score is replaced by its percentile rank inside the month
- amounts are also given relative to the month median
Columns named cat_* are integer category codes; the trainer treats them as categorical.
"""
from __future__ import annotations

import polars as pl

from .common import load_accounts, load_merchants, load_raw, save_features

ROUND_LIMITS = (500, 1000, 2000, 5000, 10000)
CATEGORICAL = ("receiver_type", "txn_type", "merchant_category", "app", "device_city")


def _codes(col: str, alias: str) -> pl.Expr:
    """Deterministic integer codes: alphabetical order of the pooled train + test values."""
    return (pl.col(col).fill_null("none").rank("dense") - 1).cast(pl.Int32).alias(alias)


def build() -> pl.DataFrame:
    df = load_raw()
    acc = load_accounts()
    mer = load_merchants()

    snd = acc.select(
        sender_id="account_id", s_home="home_city", s_bank="bank_code", s_kyc="kyc_level", s_os="os"
    )
    rcv = acc.select(
        receiver_id="account_id", r_home="home_city", r_bank="bank_code", r_kyc="kyc_level",
        r_opened="opened_on",
    )
    mrc = mer.select(receiver_id="merchant_id", m_city="city", m_onboarded="onboarded_on")
    df = (
        df.join(snd, on="sender_id", how="left", maintain_order="left")
        .join(rcv, on="receiver_id", how="left", maintain_order="left")
        .join(mrc, on="receiver_id", how="left", maintain_order="left")
    )

    amt, bal = pl.col("amount"), pl.col("balance_before")
    day = pl.col("ts").dt.date()
    feats = [
        # time
        pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("ts").dt.day().cast(pl.Int8).alias("day_of_month"),
        (pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()).cast(pl.Int16).alias("minute_of_day"),
        (pl.col("ts").dt.hour() < 6).cast(pl.Int8).alias("is_night"),
        (pl.col("ts").dt.hour() >= 20).cast(pl.Int8).alias("is_evening"),
        # amount
        amt.alias("amount"),
        amt.log1p().alias("amount_log"),
        (amt / amt.median().over("month_idx")).alias("amount_rel_month"),
        ((amt * 100).round(0).cast(pl.Int64) % 100 == 0).cast(pl.Int8).alias("amount_whole_rupee"),
        ((amt * 100).round(0).cast(pl.Int64) % 10000 == 0).cast(pl.Int8).alias("amount_round_100"),
        ((amt * 100).round(0).cast(pl.Int64) % 50000 == 0).cast(pl.Int8).alias("amount_round_500"),
        (amt % 100).alias("amount_mod_100"),
        (amt % 1000).alias("amount_mod_1000"),
        *[((amt / k).ceil() * k - amt).alias(f"gap_to_next_{k}") for k in ROUND_LIMITS],
        *[(((amt / k).ceil() * k - amt) / k).alias(f"gap_ratio_{k}") for k in ROUND_LIMITS],
        # balance
        bal.alias("balance_before"),
        (amt / (bal + 1.0)).alias("amount_over_balance"),
        (bal - amt).alias("balance_after"),
        # device, SIM, session
        pl.col("sim_age_hours"),
        pl.col("device_age_days"),
        pl.col("sender_account_age_days"),
        (pl.col("sim_age_hours") / 24.0 - pl.col("device_age_days")).alias("sim_minus_device_days"),
        (pl.col("device_age_days") / (pl.col("sender_account_age_days") + 1.0)).alias("device_age_share"),
        pl.col("new_payee").cast(pl.Int8),
        pl.col("pin_retries").cast(pl.Int8),
        pl.col("login_to_txn_sec"),
        pl.col("qr_vpa_mismatch"),
        pl.col("qr_vpa_mismatch").is_not_null().cast(pl.Int8).alias("qr_checked"),
        # ip risk: rank inside the month, never the raw value
        (pl.col("ip_risk_score").rank("average").over("month_idx") / pl.len().over("month_idx")).alias("ip_risk_rank"),
        # sender account
        (pl.col("device_city") != pl.col("s_home")).cast(pl.Int8).alias("away_from_home"),
        # receiver account (P2P) or merchant
        (day - pl.col("r_opened")).dt.total_days().alias("receiver_account_age_days"),
        (day - pl.col("m_onboarded")).dt.total_days().alias("merchant_age_days"),
        (pl.col("s_bank") == pl.col("r_bank")).cast(pl.Int8).alias("same_bank"),
        (pl.col("s_home") == pl.col("r_home")).cast(pl.Int8).alias("same_home_city"),
        (pl.col("device_city") == pl.col("r_home")).cast(pl.Int8).alias("device_in_receiver_city"),
        (pl.col("device_city") == pl.col("m_city")).cast(pl.Int8).alias("device_in_merchant_city"),
        # categoricals
        *[_codes(c, f"cat_{c}") for c in CATEGORICAL],
        _codes("s_bank", "cat_sender_bank"),
        _codes("s_kyc", "cat_sender_kyc"),
        _codes("s_os", "cat_sender_os"),
        _codes("r_kyc", "cat_receiver_kyc"),
    ]
    return df.select("txn_id", *feats)


if __name__ == "__main__":
    save_features(build(), "base_v1", "full")
