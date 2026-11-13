"""Group `labelhist_v1`: how often each entity was labelled fraud in OTHER months (research.md D5, rank 8).

    python -m src.labelhist_features          # writes labelhist_v1__full (no density view needed)

Leakage rule (research.md T4): a row never sees labels from its own month.
  - train row in Jul/Aug/Sep (or the 147 Oct-dated train rows): labels from the other months only
  - test row: labels from all train rows, including the Oct-dated train rows (research.md A5)
So the honest Sep score uses Jul+Aug labels only, exactly as the test month uses Jul-Sep labels.

Roles (an account can appear as sender or receiver, and fraud labels attach to both):
  lh_rr_*   this row's receiver, in rows where it was the receiver   (mule inflows, collusion)
  lh_rs_*   this row's receiver, in rows where it was the sender     (the mule's own onward transfers)
  lh_ss_*   this row's sender, in rows where it was the sender       (repeat refund abusers, bots)
  lh_sr_*   this row's sender, in rows where it was the receiver     (an account receiving mule money
                                                                      now sending it on)
  lh_dv_*   this row's phone                                         (device farms)
  lh_pr_*   this sender-receiver pair                                (repeat victims, collusion pairs)
Per role: rows per source month, fraud rate, and the rate of each fraud class.
Counts are divided by the number of full source months (2 for train rows, 3 for test rows).
"""
from __future__ import annotations

import polars as pl

from .common import LABELS, load_raw, save_features

FRAUD = LABELS[1:]
SHORT = {
    "mule_account": "mule", "velocity_bot": "velo", "layering_chain": "layer", "sim_swap": "sim",
    "account_takeover": "ato", "phishing_collect": "phish", "qr_swap": "qr", "refund_abuse": "refund",
    "fake_cashback": "cashback", "loan_app_extortion": "loan", "merchant_collusion": "collude",
}
ROLES = {  # role: (target key expression column, source key column)
    "rr": ("receiver_id", "receiver_id"),
    "rs": ("receiver_id", "sender_id"),
    "ss": ("sender_id", "sender_id"),
    "sr": ("sender_id", "receiver_id"),
    "dv": ("device_id", "device_id"),
    "pr": ("pair", "pair"),
}


def _source_table(src: pl.DataFrame, key: str) -> pl.DataFrame:
    """Per entity: label counts in total and per source month (month 0..3)."""
    counts = [pl.len().alias("n"), (pl.col("label") != "legit").sum().alias("f")]
    counts += [(pl.col("label") == c).sum().alias(SHORT[c]) for c in FRAUD]
    per_month = src.group_by(key, "month_idx").agg(counts)
    names = ["n", "f", *SHORT.values()]
    total = per_month.group_by(key).agg([pl.col(c).sum().alias(f"{c}__all") for c in names])
    for m in range(4):
        pm = per_month.filter(pl.col("month_idx") == m).select(key, *[pl.col(c).alias(f"{c}__m{m}") for c in names])
        total = total.join(pm, on=key, how="left")
    return total.fill_null(0)


def build() -> pl.DataFrame:
    raw = load_raw().select("txn_id", "split", "label", "month_idx", "sender_id", "receiver_id", "device_id")
    raw = raw.with_columns(pair=pl.concat_str(["sender_id", "receiver_id"], separator="|"))
    src = raw.filter(pl.col("split") == "train")
    is_train = pl.col("split") == "train"
    n_months = pl.when(is_train).then(2.0).otherwise(3.0)

    out = raw.select("txn_id")
    names = ["n", "f", *SHORT.values()]
    for role, (tkey, skey) in ROLES.items():
        table = _source_table(src, skey).rename({skey: "_k"})
        t = raw.select("txn_id", "split", "month_idx", pl.col(tkey).alias("_k")).join(
            table, on="_k", how="left", maintain_order="left"
        )
        # remove the row's own month for train rows; test rows keep everything
        own = {c: pl.coalesce([pl.when(pl.col("month_idx") == m).then(pl.col(f"{c}__m{m}")) for m in range(4)]) for c in names}
        vals = {c: (pl.col(f"{c}__all").fill_null(0) - pl.when(is_train).then(own[c]).otherwise(0).fill_null(0)) for c in names}
        n = vals["n"]
        feats = [
            (n / n_months).alias(f"lh_{role}_n_pm"),
            (vals["f"] / n_months).alias(f"lh_{role}_fraud_pm"),
            pl.when(n > 0).then(vals["f"] / n).alias(f"lh_{role}_fraud_rate"),
        ]
        feats += [pl.when(n > 0).then(vals[s] / n).alias(f"lh_{role}_r_{s}") for s in SHORT.values()]
        out = out.with_columns(t.select(feats))
    return out


if __name__ == "__main__":
    f = build()
    save_features(f, "labelhist_v1", "full")
    # leakage check: rate of the row's own class in its receiver history, train months vs test
    raw = load_raw().select("txn_id", "label", "month_idx", "split")
    d = raw.join(f.select("txn_id", "lh_rr_fraud_rate", "lh_ss_fraud_rate", "lh_dv_fraud_rate"), on="txn_id")
    print(d.group_by("split", "month_idx").agg(
        rows=pl.len(),
        rr_seen=pl.col("lh_rr_fraud_rate").is_not_null().mean().round(3),
        rr_rate=pl.col("lh_rr_fraud_rate").mean().round(4),
        ss_rate=pl.col("lh_ss_fraud_rate").mean().round(4),
        dv_rate=pl.col("lh_dv_fraud_rate").mean().round(4),
    ).sort("split", "month_idx"))
    print(d.filter(pl.col("month_idx") == 2).group_by("label").agg(
        rr=pl.col("lh_rr_fraud_rate").mean().round(3), ss=pl.col("lh_ss_fraud_rate").mean().round(3),
        dv=pl.col("lh_dv_fraud_rate").mean().round(3),
    ).sort("label"))
