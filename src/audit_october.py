"""Label-free October check (audit/gap_analysis.md sections 3 and 4).

Finds "anchor" rows with rules that need no labels (a QR paid to a person, a burst into a personal
account, a matched money hop, ...) and compares them across months:

  1. rate per 10,000 rows in the train months (chosen view) against the test month
     -> does the view give train rows the same picture the test month has?
  2. for every feature, how far the anchor rows sit from all rows, in standard deviations,
     in the train months against the test month -> which features stop working in October?

    python -m src.audit_october --view fw_s0
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from .common import ROOT, load_features, load_raw

GROUPS = ["base_v1", "flow_v1", "behaviour_v1", "behaviour_v2"]
ANCHORS = {
    "qr_to_person": pl.col("flow_qr_to_p2p") == 1,
    "p2p_burst_3plus": (pl.col("receiver_type") == "P2P") & (pl.col("rcv_in_n_both_24h") >= 2),
    "p2p_burst_6plus": (pl.col("receiver_type") == "P2P") & (pl.col("rcv_in_n_both_24h") >= 5),
    "hop_match": pl.col("rcv_next_out_ratio").is_between(0.85, 1.0) & (pl.col("rcv_next_out_secs") <= 1800),
    "hop_match_both_sides": pl.col("rcv_next_out_ratio").is_between(0.85, 1.0) & (pl.col("rcv_next_out_secs") <= 1800)
        & pl.col("snd_prev_in_ratio").is_between(0.85, 1.0) & (pl.col("snd_prev_in_secs") <= 1800),
    "reverse_pair_24h": (pl.col("pair_rev_tiny_n_past_24h") > 0) | (pl.col("pair_rev_n_future_24h") > 0),
    "device_farm_5plus_accounts": pl.col("beh_dev_senders_7d") >= 5,
    "device_burst_5_in_5min": (pl.col("beh_dev_n_past_5m") + pl.col("beh_dev_n_future_5m")) >= 4,
    "refund_3plus_in_30d": (pl.col("txn_type") == "REFUND") & ((pl.col("beh_ref_n_past_30d") + pl.col("beh_ref_n_future_30d")) >= 2),
    "new_phone_new_city": (pl.col("beh_sd_new_device") == 1) & (pl.col("beh_scity_new_city") == 1),
    "sim_changed_since_last": pl.col("beh_seq_sim_changed") == 1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="fw_s0")
    ap.add_argument("--compare", nargs="*", default=["half_s0", "full"])
    args = ap.parse_args()
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(60)
    raw = load_raw().select("txn_id", "month_idx", "receiver_type", "txn_type")
    n_month = raw.group_by("month_idx").agg(n=pl.len()).sort("month_idx")["n"].to_numpy()

    def frame(view):
        fs = [load_features(g, view).drop("txn_id") for g in GROUPS]
        return pl.concat([raw, *fs], how="horizontal")

    # ---- 1. anchor rates per view
    rates = {}
    for view in [args.view, *args.compare]:
        d = frame(view)
        for name, e in ANCHORS.items():
            c = d.filter(e.fill_null(False)).group_by("month_idx").agg(n=pl.len())
            cnt = dict(zip(c["month_idx"].to_list(), c["n"].to_list()))
            per = [cnt.get(m, 0) / n_month[m] * 1e4 for m in range(4)]
            rates[name, view] = (float(np.mean(per[:3])), per[3])
        if view != args.view:
            del d
        else:
            main_d = d
    rows = []
    for name in ANCHORS:
        r = {"anchor": name, "test": round(rates[name, args.view][1], 2)}
        for view in [args.view, *args.compare]:
            r[f"train_{view}"] = round(rates[name, view][0], 2)
            r[f"test/{view}"] = round(rates[name, view][1] / max(rates[name, view][0], 1e-9), 2)
        rows.append(r)
    print("== 1. anchor rows per 10,000 rows: train months (mean) by view, and the test month")
    print(pl.DataFrame(rows))

    # ---- 2. feature drift on the anchor rows
    d = main_d
    anchor = pl.any_horizontal([e.fill_null(False) for e in ANCHORS.values()])
    d = d.with_columns(_a=anchor)
    feats = [c for c, t in d.schema.items() if c not in ("txn_id", "month_idx", "receiver_type", "txn_type", "_a") and t.is_numeric()]
    stats = d.group_by("month_idx").agg(
        [pl.col(f).mean().alias(f"{f}|all") for f in feats] + [pl.col(f).std().alias(f"{f}|sd") for f in feats]
        + [pl.col(f).filter(pl.col("_a")).mean().alias(f"{f}|anc") for f in feats]
    ).sort("month_idx")
    out = []
    for f in feats:
        allm, sd, anc = (stats[f"{f}|{k}"].to_numpy().astype(float) for k in ("all", "sd", "anc"))
        z = (anc - allm) / np.where(sd > 0, sd, np.nan)
        zt, zo = float(np.nanmean(z[:3])), float(z[3])
        spread = float(np.nanmax(z[:3]) - np.nanmin(z[:3]))
        out.append({"feature": f, "z_train": round(zt, 3), "z_test": round(zo, 3), "change": round(zo - zt, 3),
                    "train_month_spread": round(spread, 3)})
    t = pl.DataFrame(out).with_columns(score=(pl.col("change").abs() - pl.col("train_month_spread")))
    print(f"\n== 2. anchor rows ({int(d['_a'].sum())}) against all rows, in standard deviations. "
          "'change' = test minus train; features whose change is far outside the month-to-month spread:")
    flagged = t.filter((pl.col("change").abs() > 0.25) & (pl.col("score") > 0.15)).sort(pl.col("change").abs(), descending=True)
    print(flagged.drop("score").head(60))
    (ROOT / "audit").mkdir(exist_ok=True)
    t.drop("score").sort(pl.col("change").abs(), descending=True).write_csv(ROOT / "audit" / f"october_drift__{args.view}.csv")
    print(f"all {len(feats)} features -> audit/october_drift__{args.view}.csv")


if __name__ == "__main__":
    main()
