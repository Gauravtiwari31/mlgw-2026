"""Device-id blocks: the generator hands out new phone ids from a counter, fraud first and class by class.

Facts (train, measured; audit/device_blocks.md):
  - phones 420000-422999 are the bot pool (every row on them is velocity_bot)
  - ids from 423000 to about 428445 are phones first used inside a fraud episode, laid out in label order
    (mule, velocity, layering, sim_swap, takeover, phishing, qr_swap, refund, cashback, loan, collusion);
    98% of the fraud-labelled rows on a zone's phones carry that zone's class, and the legit-labelled rest
    (about 12%) is the unreported share
  - after the block every id is an ordinary new phone, used exactly once, with almost no fraud of any kind
  - the test month is a separate run: its block starts right after the last train id

Zones are recovered WITHOUT labels for the split being processed, by an ordered segmentation (dynamic
programming) of the block's phones using three cues whose parameters come from the train block:
  1. a small row classifier (zone from row columns + the base blend's probabilities), cross-fitted in train
  2. the gap to the previous seen phone id (each zone uses its ids at a different density)
  3. whether the phone has more than one row (only sim_swap and takeover phones do)
In train the true zones come from labels (also by ordered segmentation), which gives the parameters and lets
the label-free recovery be checked.

Output: cache/device_zone.parquet (txn_id, split, zone, zone_true for train).

    python -m src.device_block --runs RUN ...      # RUN = base models (oof/ + preds/) for cue 1
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import polars as pl

from .common import CACHE, LABELS, N_CLASSES, OOF, PREDS, load_accounts, load_raw

BOT_POOL = (420000, 423000)
TRAIN_START = 423000
ZONES = LABELS[1:]
K = len(ZONES)


def block_end(ids_present: np.ndarray, rows_per_id: np.ndarray, start: int, win: int = 150) -> int:
    """First id e >= start from which three windows in a row are dense (>= 97% of ids present) and single-use."""
    hi = int(ids_present.max())
    present = np.zeros(hi - start + 2, dtype=bool)
    present[ids_present - start] = True
    multi = np.zeros(hi - start + 2, dtype=bool)
    multi[ids_present[rows_per_id > 1] - start] = True
    cs_p, cs_m = np.r_[0, np.cumsum(present)], np.r_[0, np.cumsum(multi)]

    def legit(a):
        b = min(a + win, len(present))
        return b - a >= win // 2 and (cs_p[b] - cs_p[a]) / (b - a) >= 0.97 and cs_m[b] - cs_m[a] == 0
    for e in range(0, len(present) - win):
        if legit(e) and legit(e + win) and legit(e + 2 * win):
            return start + e
    return hi + 1


def segment(score: np.ndarray) -> np.ndarray:
    """score[d, k]: fit of phone d (sorted by id) to zone k. Best non-decreasing zone assignment."""
    n, k_ = score.shape
    best = np.full((n, k_), -np.inf)
    arg = np.zeros((n, k_), dtype=np.int16)
    best[0] = score[0]
    for i in range(1, n):
        run, run_k = -np.inf, 0
        for k in range(k_):
            if best[i - 1, k] > run:
                run, run_k = best[i - 1, k], k
            best[i, k] = run + score[i, k]
            arg[i, k] = run_k
    out = np.zeros(n, dtype=np.int16)
    out[-1] = int(best[-1].argmax())
    for i in range(n - 1, 0, -1):
        out[i - 1] = arg[i, out[i]]
    return out


def row_features(raw: pl.DataFrame) -> pl.DataFrame:
    home = load_accounts().select(sender_id="account_id", s_home="home_city")
    d = raw.join(home, on="sender_id", how="left", maintain_order="left")
    return d.select(
        "txn_id",
        r_p2p=(pl.col("receiver_type") == "P2P").cast(pl.Float32),
        *[(pl.col("txn_type") == t).cast(pl.Float32).alias(f"r_{t.lower()}") for t in ("PAY", "QR", "COLLECT", "REFUND", "AUTOPAY")],
        r_amount=pl.col("amount").log1p(), r_sim=(pl.col("sim_age_hours") + 1).log(), r_dev=(pl.col("device_age_days") + 1).log(),
        r_away=(pl.col("device_city") != pl.col("s_home")).cast(pl.Float32), r_hour=pl.col("ts").dt.hour().cast(pl.Float32),
        r_newpayee=pl.col("new_payee").cast(pl.Float32), r_login=(pl.col("login_to_txn_sec") + 1).log(),
        r_pin=pl.col("pin_retries").cast(pl.Float32), r_cat=pl.col("merchant_category").fill_null("none").rank("dense").cast(pl.Float32),
        r_whole=((pl.col("amount") * 100).round(0) % 100 == 0).cast(pl.Float32),
        r_tiny=(pl.col("amount") <= 10).cast(pl.Float32),
    )


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(20)
    raw = load_raw().with_columns(dnum=pl.col("device_id").str.slice(2).cast(pl.Int64))
    feats = row_features(raw)
    order = pl.DataFrame({"zone": ZONES, "o": range(K)})

    def blend(folder):
        fr = [pl.read_parquet(folder / f"{r}.parquet") for r in args.runs]
        P = np.mean([f.select(LABELS).to_numpy() for f in fr], axis=0)
        return pl.DataFrame({"txn_id": fr[0]["txn_id"]}).with_columns([pl.Series(f"b_{c}", P[:, i].astype(np.float32)) for i, c in enumerate(LABELS)])

    def block_rows(split, folder, start):
        d = raw.filter((pl.col("split") == split) & (pl.col("dnum") >= start)).select("txn_id", "dnum", "label")
        g = d.group_by("dnum").agg(n=pl.len()).sort("dnum")
        end = block_end(g["dnum"].to_numpy(), g["n"].to_numpy(), start)
        b = d.filter(pl.col("dnum") < end).join(feats, on="txn_id", how="left").join(blend(folder), on="txn_id", how="left").sort("dnum")
        bcols = [c for c in b.columns if c.startswith("b_")]
        b = b.with_columns(pl.col(bcols).fill_null(1.0 / N_CLASSES))     # the few Oct-dated train rows have no OOF
        return b, end

    train_max = int(raw.filter(pl.col("split") == "train")["dnum"].max())
    tr, tr_end = block_rows("train", OOF, TRAIN_START)
    te, te_end = block_rows("test", PREDS, train_max + 1)
    fcols = [c for c in tr.columns if c.startswith(("r_", "b_"))]

    # ---- train: true zones from labels
    dev = tr.group_by("dnum", maintain_order=True).agg(rows=pl.len(), lab=pl.col("label").filter(pl.col("label") != "legit").mode().first(),
                                                        nf=(pl.col("label") != "legit").sum())
    S = np.zeros((dev.height, K))
    for i, (l, nf) in enumerate(zip(dev["lab"].to_list(), dev["nf"].to_list())):
        if l is not None:
            S[i, ZONES.index(l)] = nf
    z_true = segment(S)
    dev = dev.with_columns(zt=pl.Series(z_true.astype(np.int32)), gap=(pl.col("dnum") - pl.col("dnum").shift(1)).fill_null(1))
    gap_mean = np.array([max(float(dev.filter(pl.col("zt") == k)["gap"].mean() or 2.0), 1.05) for k in range(K)])
    multi = np.array([np.clip(float((dev.filter(pl.col("zt") == k)["rows"] > 1).mean() or 0.0), 0.01, 0.99) for k in range(K)])
    tr = tr.join(dev.select("dnum", "zt"), on="dnum", how="left")
    print(f"train block ids {TRAIN_START}..{tr_end - 1}: {dev.height} phones, {tr.height} rows; fraud-labelled {(tr['label'] != 'legit').mean():.3f}")
    fr = tr.filter(pl.col("label") != "legit")
    print(f"  rows whose label equals their zone, among fraud-labelled rows: {(fr['label'] == pl.Series([ZONES[k] for k in fr['zt'].to_list()])).mean():.3f}")

    # ---- cue 1: row classifier, cross-fitted in train by phone
    Xtr, ytr = tr.select(fcols).to_numpy().astype(np.float32), tr["zt"].to_numpy()
    fold = np.array([hashlib.md5(f"z:{d}".encode()).digest()[0] % 5 for d in tr["dnum"].to_list()])
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 10,
              "feature_fraction": 0.7, "lambda_l2": 2.0, "verbose": -1, "seed": 0}
    q_tr = np.zeros((tr.height, K))
    for f in range(5):
        b = lgb.train(params, lgb.Dataset(Xtr[fold != f], ytr[fold != f]), 150)
        q_tr[fold == f] = b.predict(Xtr[fold == f])
    print(f"  row classifier (cross-fitted): zone accuracy per row {np.mean(q_tr.argmax(1) == ytr):.3f}")
    q_te = lgb.train(params, lgb.Dataset(Xtr, ytr), 150).predict(te.select(fcols).to_numpy().astype(np.float32))

    prior = np.bincount(ytr, minlength=K) / len(ytr)

    def recover(b: pl.DataFrame, q: np.ndarray) -> np.ndarray:
        dn = b["dnum"].to_numpy()
        devs, inv, cnt = np.unique(dn, return_inverse=True, return_counts=True)
        E = np.zeros((len(devs), K))
        # likelihood, not posterior: divide the classifier's output by the zone's row share, or big zones
        # swallow their small neighbours
        np.add.at(E, inv, np.log(np.clip(q, 1e-3, 1.0)) - np.log(prior)[None, :])
        gap = np.r_[1, np.diff(devs)].astype(float)
        pg = 1.0 / gap_mean                                              # geometric gap model per zone
        E += np.log(pg)[None, :] + (gap[:, None] - 1) * np.log(1 - pg)[None, :]
        m = (cnt > 1)[:, None]
        E += np.where(m, np.log(multi)[None, :], np.log(1 - multi)[None, :])
        return segment(E)[inv]

    z_tr = recover(tr, q_tr)
    print(f"  label-free recovery in train: {np.mean(z_tr == ytr):.3f} of rows get their true zone")
    tr = tr.with_columns(zone=pl.Series([ZONES[k] for k in z_tr]), zone_true=pl.Series([ZONES[k] for k in ytr]))
    cmp = tr.group_by("zone_true").agg(rows=pl.len(), first_true=pl.col("dnum").min(), last_true=pl.col("dnum").max(),
                                       recovered_ok=(pl.col("zone") == pl.col("zone_true")).mean().round(3)).rename({"zone_true": "zone"})
    print(cmp.join(order, on="zone").sort("o").drop("o"))

    z_te = recover(te, q_te)
    te = te.with_columns(zone=pl.Series([ZONES[k] for k in z_te]))
    print(f"\ntest block ids {train_max + 1}..{te_end - 1}: {te['dnum'].n_unique()} phones, {te.height} rows")
    bl = te.select([f"b_{c}" for c in LABELS]).to_numpy()
    te = te.with_columns(base_pred=pl.Series([LABELS[i] for i in bl.argmax(1)]))
    zt = te.group_by("zone").agg(first_id=pl.col("dnum").min(), last_id=pl.col("dnum").max(), phones=pl.col("dnum").n_unique(), rows=pl.len(),
                                 base_legit=(pl.col("base_pred") == "legit").sum(), base_same=(pl.col("base_pred") == pl.col("zone")).sum(),
                                 base_other_fraud=((pl.col("base_pred") != "legit") & (pl.col("base_pred") != pl.col("zone"))).sum())
    print(zt.join(order, on="zone").sort("o").drop("o"))

    pool = raw.filter(pl.col("dnum").is_between(BOT_POOL[0], BOT_POOL[1] - 1)).select("txn_id", "split", zone=pl.lit("velocity_bot"), zone_true=pl.lit(None, dtype=pl.String))
    out = pl.concat([tr.select("txn_id", split=pl.lit("train"), zone="zone", zone_true="zone_true"),
                     te.select("txn_id", split=pl.lit("test"), zone="zone", zone_true=pl.lit(None, dtype=pl.String)), pool])
    CACHE.mkdir(exist_ok=True)
    out.write_parquet(CACHE / "device_zone.parquet")
    print(f"\nwrote cache/device_zone.parquet: {out.height} rows ({out.filter(pl.col('split') == 'test').height} test, incl. the bot pool)")


if __name__ == "__main__":
    main()
