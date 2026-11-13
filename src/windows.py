"""Shared window library. Every relational feature (flow, behaviour, episodes) goes through this,
so the density rule is applied the same way on every machine.

The neighbour rule (research.md D2). For a row in month m and half g, another row counts only if:
  - it is in an earlier month (any half), or
  - it is in the same month AND the same half.
Rows from later months never count. Test rows (month 3) all share one half, so they see each
other plus the Oct-dated train rows.

How it works: `build_events` gives every row a visibility class `vis` = (month, half) and adds
copies of the last days of each month into the next month's classes. After that, a plain
group-by on ['vis', key] obeys the rule, and `window_agg` / `gap_features` return one row per
original transaction.

Views: 'half_s0' (default), 'half_s1' (second seed), 'full' (all same-month rows count; only for
the density A/B test).

Example:
    ev = build_events(load_raw(), "half_s0")
    f = window_agg(ev, "receiver_id", timedelta(hours=1), "past",
                   [pl.len().alias("rcv_n_past_1h"),
                    pl.col("sender_id").n_unique().alias("rcv_senders_past_1h")])
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from .common import VIEWS, load_folds, load_raw

# First instant of months 1..3 (Aug, Sep, Oct). Month 3 also holds the few Nov-dated test rows.
MONTH_START = {1: datetime(2026, 8, 1), 2: datetime(2026, 9, 1), 3: datetime(2026, 10, 1)}
MAX_WINDOW = timedelta(days=31)


def build_events(df: pl.DataFrame, view: str, max_window: timedelta = timedelta(days=7)) -> pl.DataFrame:
    """Expand `df` (needs txn_id, ts, month_idx) so that grouping by ['vis', key] obeys the rule.

    Adds `vis` (visibility class) and `is_self` (True for the original row, False for the copies
    that make earlier-month rows visible to the next month). `max_window` is the longest look-back
    you will use on this table; it must be at most 31 days.
    """
    assert view in VIEWS, view
    assert max_window <= MAX_WINDOW, "windows longer than a month are not supported"
    folds = load_folds().select("txn_id", "half_s0", "half_s1")
    df = df.join(folds, on="txn_id", how="left", maintain_order="left")
    if view == "full":
        half = pl.lit(0, dtype=pl.Int16)
    else:
        half = pl.when(pl.col("month_idx") < 3).then(pl.col(view)).otherwise(1).cast(pl.Int16)
    base = df.with_columns(
        vis=pl.col("month_idx").cast(pl.Int16) * 2 + half, is_self=pl.lit(True)
    ).drop("half_s0", "half_s1")

    parts = [base]
    for m in (0, 1, 2):
        tail = base.filter(
            (pl.col("month_idx") == m) & (pl.col("ts") >= MONTH_START[m + 1] - max_window)
        )
        if view == "full":
            halves = [0]
        elif m + 1 == 3:
            halves = [1]
        else:
            halves = [0, 1]
        for g in halves:
            parts.append(
                tail.with_columns(vis=pl.lit((m + 1) * 2 + g, dtype=pl.Int16), is_self=pl.lit(False))
            )
    return pl.concat(parts)


def _keys(key: str | list[str]) -> list[str]:
    return [key] if isinstance(key, str) else list(key)


def window_agg(
    ev: pl.DataFrame,
    key: str | list[str],
    period: timedelta,
    direction: str,
    aggs: list[pl.Expr],
) -> pl.DataFrame:
    """Aggregate over the rows that share `key` inside a time window around each row.

    direction:
      'past'    [t - period, t)        other rows strictly before this one
      'future'  (t, t + period]        other rows strictly after this one
      'around'  [t - period, t + period]  both sides, INCLUDING this row and same-second rows
                                          (subtract the row's own contribution yourself)
    Returns txn_id plus one column per aggregate, one row per original transaction.
    Rows whose key is null are left out; a left join back on txn_id gives them nulls.
    """
    keys = _keys(key)
    by = ["vis", *keys]
    evs = ev.drop_nulls(keys).sort([*by, "ts"])
    if direction == "past":
        kw = dict(period=period, offset=-period, closed="left")
    elif direction == "future":
        kw = dict(period=period, offset=timedelta(0), closed="right")
    elif direction == "around":
        kw = dict(period=2 * period, offset=-period, closed="both")
    else:
        raise ValueError(direction)
    out = evs.rolling(index_column="ts", group_by=by, **kw).agg(aggs)
    # The result must line up row for row with the sorted input (ties share the same window).
    assert out.height == evs.height, "rolling changed the row count"
    assert out["ts"].equals(evs["ts"]), "rolling changed the row order"
    for k in by:
        assert out[k].equals(evs[k]), f"rolling changed the order of {k}"
    names = [c for c in out.columns if c not in by and c != "ts"]
    return (
        out.select(names)
        .with_columns(evs["txn_id"], evs["is_self"])
        .filter(pl.col("is_self"))
        .select("txn_id", *names)
    )


def gap_features(ev: pl.DataFrame, key: str | list[str], prefix: str) -> pl.DataFrame:
    """Seconds since the previous row and until the next row that share `key` (null if none)."""
    keys = _keys(key)
    by = ["vis", *keys]
    evs = ev.drop_nulls(keys).sort([*by, "ts"])
    out = evs.with_columns(
        (pl.col("ts") - pl.col("ts").shift(1).over(by)).dt.total_seconds().alias(f"{prefix}_secs_since_prev"),
        (pl.col("ts").shift(-1).over(by) - pl.col("ts")).dt.total_seconds().alias(f"{prefix}_secs_to_next"),
    )
    return out.filter(pl.col("is_self")).select(
        "txn_id", f"{prefix}_secs_since_prev", f"{prefix}_secs_to_next"
    )


def attach(base_ids: pl.DataFrame, *feature_frames: pl.DataFrame) -> pl.DataFrame:
    """Left-join feature frames onto the full txn_id list, keeping file order."""
    out = base_ids.select("txn_id")
    for f in feature_frames:
        out = out.join(f, on="txn_id", how="left", maintain_order="left")
    return out


if __name__ == "__main__":
    # Self-test, and a direct look at the density problem (research.md A1).
    raw = load_raw().select("txn_id", "ts", "month_idx", "split", "sender_id", "receiver_id")
    res = {}
    for view in ("full", "half_s0"):
        ev = build_events(raw, view)
        f = window_agg(
            ev, "receiver_id", timedelta(hours=24), "around",
            [(pl.len() - 1).alias("n"), pl.col("sender_id").n_unique().alias("senders")],
        )
        assert f.height == raw.height and f["txn_id"].n_unique() == raw.height
        res[view] = attach(raw, f).with_columns(raw["month_idx"], raw["split"])
        print(f"view={view}: events={ev.height}  (copies={ev.height - raw.height})")
    for view, d in res.items():
        print(view, d.group_by("month_idx").agg(
            rows=pl.len(), other_rows_same_receiver_24h=pl.col("n").mean().round(3),
            distinct_senders_24h=pl.col("senders").mean().round(3),
        ).sort("month_idx"))
