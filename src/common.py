"""Shared constants, paths and loaders. Every script imports from here.

Contracts (see work/README.md section 4):
- LABELS fixes the order of the 12 probability columns everywhere.
- month_idx: 0 = Jul, 1 = Aug, 2 = Sep, 3 = everything later (test, plus the Oct-dated train rows).
- Feature files: features/<group>_v<k>__<view>.parquet, keyed by txn_id, train and test together.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Windows consoles default to cp1252 and crash on Polars' box-drawing tables.
pl.Config.set_tbl_formatting("ASCII_FULL_CONDENSED")
pl.Config.set_tbl_rows(40)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
CACHE = ROOT / "cache"
FEATURES = ROOT / "features"
OOF = ROOT / "oof"
PREDS = ROOT / "preds"
SUBS = ROOT / "submissions"
RUNS = ROOT / "runs"
FOLDS_CSV = ROOT / "folds.csv"

LABELS = [
    "legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
    "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion",
    "merchant_collusion",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
N_CLASSES = len(LABELS)
# fw_s0 / fw_s1: "fraud whole, legit halved", the view that matches the test month
# (audit/gap_analysis.md section 3). Needs labels_clean.parquet (python -m src.labels_clean).
VIEWS = ("full", "half_s0", "half_s1", "fw_s0", "fw_s1")


def load_raw() -> pl.DataFrame:
    """Train and test stacked in file order.

    Adds: split ('train'/'test'), label (null for test), ts (Datetime), month_idx.
    Cached as parquet after the first call.
    """
    cache = CACHE / "raw.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    tr = pl.read_csv(DATA / "train.csv").with_columns(split=pl.lit("train"))
    te = pl.read_csv(DATA / "test.csv").with_columns(
        split=pl.lit("test"), label=pl.lit(None, dtype=pl.String)
    )
    df = pl.concat([tr, te.select(tr.columns)], how="vertical_relaxed")
    df = df.with_columns(ts=pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S"))
    df = df.with_columns(
        month_idx=pl.col("ts")
        .dt.strftime("%Y-%m")
        .replace_strict({"2026-07": 0, "2026-08": 1, "2026-09": 2}, default=3, return_dtype=pl.Int8)
    )
    CACHE.mkdir(exist_ok=True)
    df.write_parquet(cache)
    return df


def load_accounts() -> pl.DataFrame:
    return pl.read_csv(DATA / "accounts.csv", try_parse_dates=True)


def load_merchants() -> pl.DataFrame:
    return pl.read_csv(DATA / "merchants.csv", try_parse_dates=True)


def load_folds() -> pl.DataFrame:
    if not FOLDS_CSV.exists():
        raise FileNotFoundError("folds.csv is missing. Run: python -m src.folds")
    return pl.read_csv(FOLDS_CSV)


def feature_path(group: str, view: str) -> Path:
    """Path of a feature file. `group` includes its version, e.g. 'flow_v1'."""
    assert view in VIEWS, view
    return FEATURES / f"{group}__{view}.parquet"


def save_features(df: pl.DataFrame, group: str, view: str) -> Path:
    """Write a feature file after checking the contract: one row per txn_id, all rows present."""
    n = load_raw().height
    assert "txn_id" in df.columns, "feature file needs txn_id"
    assert df.height == n, f"{group}: {df.height} rows, expected {n}"
    assert df["txn_id"].n_unique() == n, f"{group}: duplicate txn_id"
    banned = {"label", "app_version", "ip_risk_score"} & set(df.columns)
    assert not banned, f"{group}: banned raw columns {banned} (research.md T2, T4)"
    FEATURES.mkdir(exist_ok=True)
    path = feature_path(group, view)
    df.write_parquet(path)
    print(f"wrote {path.relative_to(ROOT)}  rows={df.height}  features={df.width - 1}")
    return path


def load_features(group: str, view: str) -> pl.DataFrame:
    """Load a group in the requested view, falling back to the 'full' file for single-row groups."""
    path = feature_path(group, view)
    if not path.exists():
        path = feature_path(group, "full")
    if not path.exists():
        raise FileNotFoundError(f"no feature file for {group} (view {view})")
    return pl.read_parquet(path)
