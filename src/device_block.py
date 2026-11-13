"""Device-id blocks: the generator hands out new phone ids from a counter, fraud first and class by class.

Facts (train, measured; audit/device_blocks.md):
  - phones 420000-422999 are the bot pool (every row on them is velocity_bot)
  - ids from 423000 to 428454 are phones first used inside a fraud episode, laid out in label order
    (mule, velocity, layering, sim_swap, takeover, phishing, qr_swap, refund, cashback, loan, collusion);
    98% of the fraud-labelled rows on a zone's phones carry that zone's class, and the legit-labelled rest
    (about 12%) is the unreported share
  - after the block every id is an ordinary new phone, used exactly once with no id skipped, and with almost
    no fraud of any kind. So the block ends right after the last id that no row uses.
  - the test month is a separate run: its block starts right after the last train id
  - zone lengths (in ids) of the test block follow the train block's proportions closely

Zones are recovered WITHOUT labels for the split being processed, by an ordered segmentation (dynamic
programming) of the block's phones using three cues whose parameters come from the train block:
  1. a small row classifier (zone from row columns + the base blend's probabilities), cross-fitted in train
  2. the gap to the previous seen phone id (each zone uses its ids at a different density)
  3. whether the phone has more than one row (only sim_swap and takeover phones do)
In train the true zones come from labels (also by ordered segmentation), which gives the parameters and lets
the label-free recovery be checked.

The test block is 6 times shorter than the train block, so its boundaries are less certain. For test the
segmentation runs over ids (not phones) with a prior on each zone's length (train length scaled to the block,
negative binomial). `--simulate N` checks this on N test-sized blocks cut out of the train block.

Output: cache/device_zone.parquet (txn_id, split, zone, zone_true for train).

    python -m src.device_block --runs RUN ...      # RUN = base models (oof/ + preds/) for cue 1
    python -m src.device_block --runs RUN ... --simulate 300
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


def block_end(ids_present: np.ndarray, start: int) -> int:
    """Ordinary new phones use every id exactly once, so the block ends right after the last id no row uses."""
    ids = np.unique(ids_present[ids_present >= start])
    miss = np.setdiff1d(np.arange(start, int(ids.max()) + 1), ids)
    return int(miss.max()) + 1 if len(miss) else start


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


def length_prior(mu: np.ndarray, n: int, v: float) -> np.ndarray:
    """log P(zone k is L ids long), L = 0..n. Mean mu[k], variance v * mean (v = 1: Poisson)."""
    from scipy.stats import nbinom, poisson
    L = np.arange(n + 1)[None, :]
    if v <= 1.0:
        return poisson.logpmf(L, mu[:, None])
    return nbinom.logpmf(L, (mu / (v - 1.0))[:, None], 1.0 / v)


def segment_ids(E: np.ndarray, lp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """E[i, k]: log-fit of id i (ids in order, seen or not) to zone k. Zones are consecutive id intervals in
    label order; lp[k, L] is the log prior of zone k being L ids long (None = flat).
    Returns (best zone per id, P(zone | everything) per id)."""
    from scipy.special import logsumexp
    n, k_ = E.shape
    C = np.vstack([np.zeros((1, k_)), np.cumsum(E, axis=0)])
    pos = np.arange(n + 1)
    Ld = pos[None, :] - pos[:, None]                                    # [a, b] -> b - a
    ok = Ld >= 0
    Lc = np.clip(Ld, 0, n)

    def seg_(k):                                                         # score of zone k covering [a, b)
        m = C[None, :, k] - C[:, None, k]
        if lp is not None:
            m = m + lp[k][Lc]
        return np.where(ok, m, -np.inf)
    segs = [seg_(k) for k in range(k_)]
    seg = segs.__getitem__

    first = np.full(n + 1, -np.inf); first[0] = 0.0
    Fm, Fs, arg = [first], [first], [None]
    for k in range(k_):
        S = seg(k)
        M = Fm[-1][:, None] + S
        arg.append(M.argmax(0)); Fm.append(M.max(0))
        Fs.append(logsumexp(Fs[-1][:, None] + S, axis=0))
    z = np.zeros(n, dtype=np.int16)
    b = n
    for k in range(k_, 0, -1):
        a = int(arg[k][b]); z[a:b] = k - 1; b = a
    last = np.full(n + 1, -np.inf); last[n] = 0.0
    Bs = [last]
    for k in range(k_ - 1, -1, -1):
        Bs.append(logsumexp(seg(k) + Bs[-1][None, :], axis=1))
    Bs = Bs[::-1]                                                        # Bs[k][a]: zones k.. cover [a, n)
    Z = Bs[0][0]
    P = np.zeros((n, k_))
    for k in range(k_):
        G = np.exp(Fs[k][:, None] + seg(k) + Bs[k + 1][None, :] - Z)    # zone k is exactly [a, b)
        R = np.cumsum(G[:, ::-1], axis=1)[:, ::-1]                       # sum over b >= j
        T = np.cumsum(R, axis=0)                                         # sum over a <= i
        P[:, k] = T[pos[:-1], pos[1:]]
    return z, P


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
    ap.add_argument("--simulate", type=int, default=0, help="check the test-block method on this many test-sized blocks cut from train")
    ap.add_argument("--dispersion", type=float, default=1.0, help="zone-length prior: variance / mean (1 = Poisson; 0 = no prior)")
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
        end = block_end(g["dnum"].to_numpy(), start)
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

    # ---- test block: segmentation over ids, with a prior on zone lengths (see the module docstring)
    first = np.array([int(dev.filter(pl.col("zt") >= k)["dnum"].min()) for k in range(K)])     # first id of each true train zone
    first[0] = TRAIN_START
    span = np.diff(np.r_[first, tr_end]).astype(float)
    seen = np.clip(np.array([dev.filter(pl.col("zt") == k).height for k in range(K)]) / span, 0.02, 0.98)
    zone_of_id = np.repeat(np.arange(K), span.astype(int))                                     # true zone of every train block id

    def emissions(b: pl.DataFrame, q: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        n = end - start
        idx = b["dnum"].to_numpy() - start
        rows = np.zeros((n, K))
        np.add.at(rows, idx, np.log(np.clip(q, 1e-3, 1.0)) - np.log(prior)[None, :])
        cnt = np.bincount(idx, minlength=n)
        E = np.tile(np.log(1 - seen)[None, :], (n, 1))                                          # an id no row uses
        s_ = cnt > 0
        E[s_] = rows[s_] + np.log(seen)[None, :] + np.where((cnt[s_] > 1)[:, None], np.log(multi)[None, :], np.log(1 - multi)[None, :])
        return E, cnt

    def prior_for(n: int, v: float):
        return None if v <= 0 else length_prior(span * n / span.sum(), n, v)

    if args.simulate:
        E_full, cnt_full = emissions(tr, q_tr, TRAIN_START, tr_end)
        n_te = te_end - (train_max + 1)
        rng = np.random.default_rng(0)
        variants = {"no length prior": 0.0, "prior, variance = mean (Poisson)": 1.0, "prior, variance = 3 x mean": 3.0}
        print(f"\nsimulation: {args.simulate} blocks of about {n_te} ids cut from the train block (classifier cross-fitted)")
        for v_sim, sim_name in ((1.0, "zone lengths vary like Poisson"), (4.0, "zone lengths vary 2x more than Poisson")):
            acc = {k: [] for k in variants}; accm = {k: [] for k in variants}; per = {k: np.zeros((K, 2)) for k in variants}
            for _ in range(args.simulate):
                parts, truth = [], []
                for k in range(K):
                    mu = span[k] * n_te / span.sum()
                    l = rng.poisson(mu) if v_sim <= 1 else rng.negative_binomial(mu / (v_sim - 1), 1 / v_sim)
                    l = int(np.clip(l, 1, span[k]))
                    a = int(first[k] - TRAIN_START + rng.integers(0, span[k] - l + 1))
                    parts.append(np.arange(a, a + l)); truth.append(np.full(l, k))
                ii, tt = np.concatenate(parts), np.concatenate(truth)
                E, w = E_full[ii], cnt_full[ii]
                for name, v in variants.items():
                    z, P = segment_ids(E, prior_for(len(ii), v))
                    acc[name].append(np.sum(w * (z == tt)) / w.sum()); accm[name].append(np.sum(w * (P.argmax(1) == tt)) / w.sum())
                    for k in range(K):
                        per[name][k] += (np.sum(w * ((z == tt) & (tt == k))), np.sum(w * (tt == k)))
            print(f"  {sim_name}:")
            for name in variants:
                a = np.array(acc[name])
                print(f"    {name:36s} rows in the right zone: mean {a.mean():.3f}, worst 10% of blocks {np.quantile(a, 0.1):.3f}; by marginals {np.mean(accm[name]):.3f}"
                      f"   | per zone " + " ".join(f"{ZONES[k][:5]} {per[name][k, 0] / per[name][k, 1]:.2f}" for k in range(K)))

    E_te, cnt_te = emissions(te, q_te, train_max + 1, te_end)
    z_ids, P_ids = segment_ids(E_te, prior_for(len(E_te), args.dispersion))
    idx_te = te["dnum"].to_numpy() - (train_max + 1)
    z_te = z_ids[idx_te]
    z_old = recover(te, q_te)
    print(f"\ntest block: zone-length prior with variance = {args.dispersion} x mean; rows whose zone differs from the phone-space "
          f"segmentation without a prior: {int((z_te != z_old).sum())} of {te.height}")
    te = te.with_columns(zone=pl.Series([ZONES[k] for k in z_te]), p_zone=pl.Series(P_ids[idx_te, z_te].astype(np.float32)))
    exp_len = span * len(E_te) / span.sum()
    got_len = np.bincount(z_ids, minlength=K)
    print("  zone lengths in ids, expected from train proportions vs found: " + ", ".join(f"{ZONES[k][:6]} {exp_len[k]:.0f}/{got_len[k]}" for k in range(K)))
    print(f"  rows with zone probability under 0.8: {int((te['p_zone'] < 0.8).sum())}; under 0.6: {int((te['p_zone'] < 0.6).sum())}")
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
    out = out.join(te.select("txn_id", "p_zone"), on="txn_id", how="left", maintain_order="left")
    CACHE.mkdir(exist_ok=True)
    out.write_parquet(CACHE / "device_zone.parquet")
    print(f"\nwrote cache/device_zone.parquet: {out.height} rows ({out.filter(pl.col('split') == 'test').height} test, incl. the bot pool)")


if __name__ == "__main__":
    main()
