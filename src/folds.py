"""Build folds.csv. Deterministic: every machine gets the identical file from the same data.

The half flags come from a hash of txn_id, not from a random generator, so they do not depend
on row order, library versions or the operating system. Compare the printed checksum in the
team chat after running. Never edit folds.csv by hand.

Columns:
  txn_id      every train and test row
  split       train / test
  month_fold  0 = Jul, 1 = Aug, 2 = Sep, -1 = Oct-dated train rows and all test rows
  is_sep_val  1 for Sep train rows
  half_s0/1   0 or 1 per Jul-Sep train row; 1 for every other row (research.md D2)
"""
from __future__ import annotations

import hashlib

import polars as pl

from .common import FOLDS_CSV, load_raw


def _half(txn_ids: list[str], seed: int) -> list[int]:
    return [hashlib.md5(f"{seed}:{t}".encode()).digest()[0] & 1 for t in txn_ids]


def main() -> None:
    df = load_raw().select("txn_id", "split", "month_idx")
    in_cv = (pl.col("split") == "train") & (pl.col("month_idx") < 3)
    ids = df["txn_id"].to_list()
    df = df.with_columns(
        month_fold=pl.when(in_cv).then(pl.col("month_idx")).otherwise(-1).cast(pl.Int8),
        h0=pl.Series(_half(ids, 0), dtype=pl.Int8),
        h1=pl.Series(_half(ids, 1), dtype=pl.Int8),
    ).with_columns(
        is_sep_val=(pl.col("month_fold") == 2).cast(pl.Int8),
        half_s0=pl.when(in_cv).then(pl.col("h0")).otherwise(1).cast(pl.Int8),
        half_s1=pl.when(in_cv).then(pl.col("h1")).otherwise(1).cast(pl.Int8),
    )
    out = df.select("txn_id", "split", "month_fold", "is_sep_val", "half_s0", "half_s1")
    out.write_csv(FOLDS_CSV)

    print(out.group_by("split", "month_fold").agg(
        rows=pl.len(), half_s0=pl.col("half_s0").mean().round(4), half_s1=pl.col("half_s1").mean().round(4)
    ).sort("split", "month_fold"))
    digest = hashlib.sha256(FOLDS_CSV.read_bytes()).hexdigest()
    print(f"wrote {FOLDS_CSV.name}  rows={out.height}  sha256={digest[:16]}")


if __name__ == "__main__":
    main()
