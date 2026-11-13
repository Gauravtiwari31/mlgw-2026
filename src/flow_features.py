"""Group `flow_v1`: features that follow the money (work/L2 tasks 1-3).

    python -m src.flow_features                  # view half_s0 (default)
    python -m src.flow_features --view half_s1

Blocks:
  A  rcv_in_*    other payments INTO this row's receiver, before and after the row (fan-in; mule)
     rcv_qr_*    QR payments and QR mismatches at the receiver (qr_swap)
  B  rcv_out_*   payments the receiver itself SENDS around the row (pass-through; mule, layering)
     snd_in_*    payments the sender RECEIVED around the row (onward transfer; mule, layering)
  C  pair_rev_*  payments in the opposite direction between the same two accounts
                 (a small credit first, then a big payment back; fake_cashback)

Every window goes through src/windows.py, so the density rule is applied. 'past' windows are
[t - w, t) and 'future' windows are (t, t + w]: the row itself is never counted.
Receiver account age and KYC are already in base_v1 and are not repeated here.
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import polars as pl

from .common import load_accounts, load_raw, save_features
from .windows import attach, build_events, window_agg

W = {
    "1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1), "6h": timedelta(hours=6), "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}
TINY = 10.0  # rupees: the "small credit to look genuine" in fake_cashback
ASOF_TOLERANCE = timedelta(days=7)


def _load() -> pl.DataFrame:
    """Raw rows with integer ids (faster grouping) and the few flags the windows need."""
    raw = load_raw().select(
        "txn_id", "ts", "month_idx", "sender_id", "receiver_id", "receiver_type", "txn_type",
        "amount", "new_payee", "qr_vpa_mismatch",
    )
    ids = (
        pl.concat([raw["sender_id"], raw["receiver_id"]]).unique().sort().to_frame("id").with_row_index("code")
    )
    home = load_accounts().select(sender_id="account_id", s_home="home_city")
    cities = home.select(pl.col("s_home").unique().sort()).with_row_index("s_city")
    home = home.join(cities, on="s_home", how="left").select("sender_id", "s_city")
    return (
        raw.join(ids.rename({"id": "sender_id", "code": "snd"}), on="sender_id", how="left", maintain_order="left")
        .join(ids.rename({"id": "receiver_id", "code": "rcv"}), on="receiver_id", how="left", maintain_order="left")
        .join(home, on="sender_id", how="left", maintain_order="left")
        .with_columns(
            is_p2p=(pl.col("receiver_type") == "P2P"),
            is_qr=(pl.col("txn_type") == "QR").cast(pl.Int32),
            qr_mm=(pl.col("qr_vpa_mismatch") == 1).fill_null(False).cast(pl.Int32),
            new_payee=pl.col("new_payee").cast(pl.Float32),
        )
        .select("txn_id", "ts", "month_idx", "snd", "rcv", "s_city", "is_p2p", "is_qr", "qr_mm",
                "amount", "new_payee")
    )


def _nearest(ev: pl.DataFrame, key: str, self_mask: pl.Expr, other_mask: pl.Expr, strategy: str,
             prefix: str) -> pl.DataFrame:
    """Nearest earlier ('backward') or later ('forward') OTHER event with the same key.

    Returns txn_id, <prefix>_secs (time gap, always positive) and <prefix>_amt (its amount).
    Works inside a visibility class, so the density rule holds.
    """
    left = (
        ev.filter(pl.col("is_self") & self_mask).drop_nulls(key)
        .select("txn_id", "vis", key, "ts").sort("ts")
    )
    right = (
        ev.filter(other_mask).drop_nulls(key)
        .select("vis", key, o_ts="ts", o_amt="amount").sort("o_ts")
    )
    j = left.join_asof(
        right, left_on="ts", right_on="o_ts", by=["vis", key], strategy=strategy,
        allow_exact_matches=False, tolerance=ASOF_TOLERANCE, check_sortedness=False,
    )
    return j.select(
        "txn_id",
        (pl.col("o_ts") - pl.col("ts")).dt.total_seconds().abs().alias(f"{prefix}_secs"),
        pl.col("o_amt").alias(f"{prefix}_amt"),
    )


def block_receiver_inflow(raw: pl.DataFrame, view: str) -> list[pl.DataFrame]:
    """A: other payments into the same receiver (accounts and merchants)."""
    ev = build_events(raw, view, max_window=W["7d"])
    frames = []
    for d in ("past", "future"):
        for w in ("1m", "5m", "30m", "1h", "6h", "24h", "7d"):
            t = f"{d}_{w}"
            aggs = [pl.len().alias(f"rcv_in_n_{t}")]
            if w != "1m":
                aggs.append(pl.col("amount").sum().alias(f"rcv_in_amt_{t}"))
            if w in ("30m", "1h", "6h", "24h", "7d"):
                aggs.append(pl.col("snd").n_unique().alias(f"rcv_in_senders_{t}"))
            if w in ("1h", "24h", "7d"):
                aggs.append(pl.col("new_payee").mean().alias(f"rcv_in_newpayee_share_{t}"))
            if w in ("1h", "24h"):
                aggs.append(pl.col("s_city").n_unique().alias(f"rcv_in_cities_{t}"))
            if w in ("24h", "7d"):
                aggs.append(pl.col("is_qr").sum().alias(f"rcv_qr_n_{t}"))
                aggs.append(pl.col("qr_mm").sum().alias(f"rcv_qr_mm_{t}"))
            frames.append(window_agg(ev, "rcv", W[w], d, aggs))
    return frames


def block_pass_through(raw: pl.DataFrame, view: str) -> list[pl.DataFrame]:
    """B: one event table per ACCOUNT holding both its inbound and its outbound payments.

    Only person-to-person payments create an inbound event, because merchants never send money.
    """
    cols = ["txn_id", "ts", "month_idx", "amount"]
    inbound = raw.filter(pl.col("is_p2p")).select(*cols, acct="rcv", other="snd", is_out=pl.lit(False))
    outbound = raw.select(*cols, acct="snd", other="rcv", is_out=pl.lit(True))
    ev = build_events(pl.concat([inbound, outbound]), view, max_window=W["7d"])
    out, inn = pl.col("is_out"), ~pl.col("is_out")
    # The row as a payment INTO its receiver: what does the receiver send out around it?
    ev_rcv = ev.with_columns(is_self=pl.col("is_self") & inn)
    # The row as a payment OUT OF its sender: what did the sender receive around it?
    ev_snd = ev.with_columns(is_self=pl.col("is_self") & out)

    frames = []
    for w in ("5m", "30m", "1h", "6h", "24h", "7d"):
        aggs = [out.sum().alias(f"rcv_out_n_future_{w}"),
                pl.col("amount").filter(out).sum().alias(f"rcv_out_amt_future_{w}")]
        if w in ("1h", "24h"):
            aggs.append(pl.col("other").filter(out).n_unique().alias(f"rcv_out_receivers_future_{w}"))
        frames.append(window_agg(ev_rcv, "acct", W[w], "future", aggs))

        aggs = [inn.sum().alias(f"snd_in_n_past_{w}"),
                pl.col("amount").filter(inn).sum().alias(f"snd_in_amt_past_{w}")]
        if w in ("1h", "24h"):
            aggs.append(pl.col("other").filter(inn).n_unique().alias(f"snd_in_senders_past_{w}"))
        frames.append(window_agg(ev_snd, "acct", W[w], "past", aggs))
    for w in ("1h", "24h", "7d"):
        frames.append(window_agg(ev_rcv, "acct", W[w], "past", [
            out.sum().alias(f"rcv_out_n_past_{w}"),
            pl.col("amount").filter(out).sum().alias(f"rcv_out_amt_past_{w}")]))
        frames.append(window_agg(ev_snd, "acct", W[w], "future", [
            inn.sum().alias(f"snd_in_n_future_{w}"),
            pl.col("amount").filter(inn).sum().alias(f"snd_in_amt_future_{w}")]))

    frames.append(_nearest(ev, "acct", inn, out, "forward", "rcv_next_out"))
    frames.append(_nearest(ev, "acct", inn, out, "backward", "rcv_prev_out"))
    frames.append(_nearest(ev, "acct", out, inn, "backward", "snd_prev_in"))
    frames.append(_nearest(ev, "acct", out, inn, "forward", "snd_next_in"))
    return frames


def block_reverse_pair(raw: pl.DataFrame, view: str) -> list[pl.DataFrame]:
    """C: payments going the other way between the same two accounts."""
    cols = ["txn_id", "ts", "month_idx", "amount"]
    n_ids = int(max(raw["snd"].max(), raw["rcv"].max())) + 1
    pair = lambda a, b: (pl.col(a).cast(pl.Int64) * n_ids + pl.col(b).cast(pl.Int64)).alias("pair")  # noqa: E731
    fwd = raw.select(*cols, pair("snd", "rcv"), is_rev=pl.lit(False))
    rev = raw.filter(pl.col("is_p2p")).select(*cols, pair("rcv", "snd"), is_rev=pl.lit(True))
    ev = build_events(pl.concat([fwd, rev]), view, max_window=W["7d"])
    ev = ev.with_columns(is_self=pl.col("is_self") & ~pl.col("is_rev"))
    r = pl.col("is_rev")
    tiny = r & (pl.col("amount") <= TINY)

    frames = []
    for w in ("1h", "24h", "7d"):
        frames.append(window_agg(ev, "pair", W[w], "past", [
            r.sum().alias(f"pair_rev_n_past_{w}"),
            tiny.sum().alias(f"pair_rev_tiny_n_past_{w}"),
            pl.col("amount").filter(r).sum().alias(f"pair_rev_amt_past_{w}")]))
        frames.append(window_agg(ev, "pair", W[w], "future", [
            r.sum().alias(f"pair_rev_n_future_{w}"),
            pl.col("amount").filter(r).sum().alias(f"pair_rev_amt_future_{w}")]))
    frames.append(_nearest(ev, "pair", ~r, tiny, "backward", "pair_rev_tiny_prev"))
    frames.append(_nearest(ev, "pair", ~r, r, "backward", "pair_rev_prev"))
    frames.append(_nearest(ev, "pair", ~r, r, "forward", "pair_rev_next"))
    return frames


def build(view: str) -> pl.DataFrame:
    t0 = time.time()
    raw = _load()
    frames = []
    for name, block in (("A receiver inflow", block_receiver_inflow), ("B pass-through", block_pass_through),
                        ("C reverse pair", block_reverse_pair)):
        frames += block(raw, view)
        print(f"  block {name} done ({time.time() - t0:.0f}s)", flush=True)
    f = attach(raw, *frames).with_columns(raw["amount"], raw["is_p2p"], raw["is_qr"])

    amt = pl.col("amount")
    eps = 1.0
    derived = [
        # QR scanned but the money goes to a person, not a shop (qr_swap)
        ((pl.col("is_qr") == 1) & pl.col("is_p2p")).cast(pl.Int8).alias("flow_qr_to_p2p"),
    ]
    for w in ("30m", "6h", "24h", "7d"):
        n = pl.col(f"rcv_in_n_past_{w}") + pl.col(f"rcv_in_n_future_{w}")
        s = pl.col(f"rcv_in_senders_past_{w}") + pl.col(f"rcv_in_senders_future_{w}")
        a = pl.col(f"rcv_in_amt_past_{w}") + pl.col(f"rcv_in_amt_future_{w}")
        derived += [
            n.alias(f"rcv_in_n_both_{w}"),
            (s / n).alias(f"rcv_in_senders_per_payment_{w}"),      # 1.0 = every payment from a new person
            (amt / (a / n)).alias(f"rcv_amount_over_in_mean_{w}"),  # this payment vs the receiver's usual
        ]
    for w in ("24h", "7d"):
        qr = pl.col(f"rcv_qr_n_past_{w}") + pl.col(f"rcv_qr_n_future_{w}")
        n = pl.col(f"rcv_in_n_past_{w}") + pl.col(f"rcv_in_n_future_{w}")
        mm = pl.col(f"rcv_qr_mm_past_{w}") + pl.col(f"rcv_qr_mm_future_{w}")
        derived += [
            ((qr + pl.col("is_qr")) / (n + 1)).alias(f"rcv_qr_share_{w}"),
            mm.alias(f"rcv_qr_mm_both_{w}"),
            (mm / qr).alias(f"rcv_qr_mm_share_{w}"),
        ]
    for w in ("30m", "1h", "6h", "24h", "7d"):
        derived += [
            # money the receiver sends on afterwards, against what it had just taken in
            (pl.col(f"rcv_out_amt_future_{w}") / amt).alias(f"rcv_out_over_amount_{w}"),
            (pl.col(f"rcv_out_amt_future_{w}") / (pl.col(f"rcv_in_amt_past_{w}") + amt + eps)).alias(f"rcv_out_over_in_{w}"),
            # this payment against what its sender had just received
            (amt / (pl.col(f"snd_in_amt_past_{w}") + eps)).alias(f"snd_amount_over_in_{w}"),
        ]
    derived += [
        (pl.col("rcv_next_out_amt") / amt).alias("rcv_next_out_ratio"),          # next hop keeps 0.85-1.0x
        (amt / pl.col("snd_prev_in_amt")).alias("snd_prev_in_ratio"),            # previous hop
        (pl.col("snd_next_in_amt") / amt).alias("snd_next_in_ratio"),            # small credit, big return
        (amt / pl.col("pair_rev_tiny_prev_amt")).alias("pair_amount_over_rev_tiny"),
        (amt / pl.col("pair_rev_prev_amt")).alias("pair_amount_over_rev_prev"),
        (pl.col("pair_rev_next_amt") / amt).alias("pair_rev_next_ratio"),
    ]
    f = f.with_columns(derived).drop("amount", "is_p2p", "is_qr")
    # ratios with a zero denominator are "not applicable", not infinitely large
    num = [c for c, t in f.schema.items() if t.is_float()]
    f = f.with_columns(pl.when(pl.col(num).is_finite()).then(pl.col(num)).otherwise(None).name.keep())
    f = f.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    print(f"  {f.width - 1} features in {time.time() - t0:.0f}s")
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="half_s0", choices=["half_s0", "half_s1", "full", "fw_s0", "fw_s1"])
    a = ap.parse_args()
    save_features(build(a.view), "flow_v1", a.view)
