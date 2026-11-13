"""Build labels_clean.parquet: which train labels are real (audit/gap_analysis.md, section 1).

Train labels have three kinds of error:
  noise       an ordinary legit row carrying a fraud label (about 63% of all fraud labels)
  unreported  a row that looks exactly like fraud but is labelled legit
  wrong type  real fraud filed under another fraud type

Method: a cross-fitted 12-class LightGBM (4 random folds, so no row is scored by a model that
saw its own label) on every feature we have at FULL density, plus raw ip_risk_score and
app_version. Those two columns are banned as model features because they stop working in
October, but inside the train months they mark real fraud very well, and this file only
decides which train labels to believe. Nothing here touches a test row.

Output columns (one row per train row):
  txn_id, label_clean, keep, status, p_fraud, top_fraud, top_share
  keep = False: do not train on the row and do not score it (ambiguous / unreported / wrong type)

    python -m src.labels_clean --from-changes   # every machine: rebuild from the committed list
    python -m src.labels_clean            # refit rules from cache/labels_clean_probs.parquet
    python -m src.labels_clean --refit    # refit the model too (8 min; needs the *_full feature files)
"""
from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import polars as pl

from .common import CACHE, LABEL_TO_ID, LABELS, N_CLASSES, ROOT, load_features, load_raw  # noqa: F401

GROUPS = ["base_v1", "flow_v1", "behaviour_v1", "behaviour_v2"]
PROBS = CACHE / "labels_clean_probs.parquet"
OUT = ROOT / "labels_clean.parquet"
NOISE_BELOW = 0.10        # fraud label, P(fraud) under this: an ordinary row, becomes legit
REAL_FROM = 0.30          # fraud label, P(fraud) from this: real fraud
UNREPORTED_FROM = 0.50    # legit label, P(fraud) from this: unreported fraud
LEGIT_KEEP = 0.25         # share of legit-labelled rows used to fit each fold (weight 1 / share)


def crossfit() -> pl.DataFrame:
    import lightgbm as lgb

    raw = load_raw()
    is_tr = (raw["split"] == "train").to_numpy()
    feats = [load_features(g, "full") for g in GROUPS]
    for f in feats:
        assert f["txn_id"].equals(raw["txn_id"])
    names = [c for f in feats for c in f.columns if c != "txn_id"] + ["ip_risk_score", "app_version"]
    X = np.hstack(
        [f.drop("txn_id").select(pl.all().cast(pl.Float32)).to_numpy()[is_tr] for f in feats]
        + [raw.select(pl.col("ip_risk_score", "app_version").cast(pl.Float32)).to_numpy()[is_tr]]
    )
    del feats
    ids = raw["txn_id"].filter(pl.Series(is_tr))
    y = raw["label"].filter(pl.Series(is_tr)).replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    h = np.array([hashlib.md5(f"clean:{t}".encode()).digest()[:2] for t in ids.to_list()], dtype="S2")
    hb = np.frombuffer(h.tobytes(), dtype=np.uint8).reshape(-1, 2)
    fold, pick = hb[:, 0] % 4, hb[:, 1]
    use = (y != 0) | (pick < int(256 * LEGIT_KEEP))
    w = np.where(y == 0, 1.0 / LEGIT_KEEP, 1.0).astype(np.float32)
    cat = [n for n in names if n.startswith("cat_")]
    params = {"objective": "multiclass", "num_class": N_CLASSES, "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "max_bin": 255, "verbose": -1, "seed": 0}
    P = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    t0 = time.time()
    for k in range(4):
        tr = np.where(use & (fold != k))[0]
        b = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr], feature_name=names, categorical_feature=cat), 250)
        va = np.where(fold == k)[0]
        P[va] = b.predict(X[va])
        print(f"  fold {k}: {len(tr)} rows, {time.time() - t0:.0f}s", flush=True)
    out = pl.DataFrame({"txn_id": ids}).with_columns(
        [pl.Series(LABELS[c], P[:, c], dtype=pl.Float32) for c in range(N_CLASSES)])
    CACHE.mkdir(exist_ok=True)
    out.write_parquet(PROBS)
    return out


CHANGES = ROOT / "audit" / "labels_clean_changes.csv"   # in git: every row whose label is not taken as is


def from_changes() -> None:
    """Rebuild labels_clean.parquet from the committed change list, with no model fit.
    This is what every other machine should run, so all machines use identical labels."""
    raw = load_raw().filter(pl.col("split") == "train").select("txn_id", "label")
    ch = pl.read_csv(CHANGES)
    out = raw.join(ch, on="txn_id", how="left", maintain_order="left").select(
        "txn_id",
        label_clean=pl.col("label_clean").fill_null(pl.col("label")),
        keep=pl.col("keep").fill_null(True),
        status=pl.col("status").fill_null(pl.when(pl.col("label") == "legit").then(pl.lit("legit")).otherwise(pl.lit("fraud"))),
    )
    out.write_parquet(OUT)
    print(f"wrote {OUT.name} from {CHANGES.name}: {ch.height} changed rows, "
          f"clean fraud share {out.filter(pl.col('keep'))['label_clean'].ne('legit').mean():.2%}")


TYPE_RULES = False  # OFF: tested on the public board 13 Nov, 0.4738 against 0.4987 without them


def apply_type_rules(out: pl.DataFrame) -> pl.DataFrame:
    """EXPERIMENT, switched off (TYPE_RULES = False). Public score fell 0.4987 -> 0.4738 with these rules.
    Rule 4 (qr_swap to a merchant -> collusion) explains most of the loss: in the clean test labels
    those rows are qr_swap. The other rules moved too few test rows to judge.

    Fix fraud rows filed under the wrong type, using each class's definition (PROBLEM_STATEMENT.md).

    Only rows already judged real fraud are touched. Status of a changed row becomes 'type_rule'.
      1. refund_abuse rows that are not refund claims, from senders who never claim a refund
         -> merchant_collusion (round-number sales to one merchant from a few accounts)
      2. sim_swap / account_takeover, per sender-day episode:
           fresh SIM (< 72 h) and no new phone (< 3 days) -> sim_swap
           new phone and no fresh SIM                    -> account_takeover
      3. phishing_collect rows that are part of a fake-cashback exchange (a <= Rs 10 credit between
         the pair followed by a payment back within 24 h, either direction)  -> fake_cashback
         fake_cashback rows that are COLLECT requests with no such exchange   -> phishing_collect
      4. qr_swap rows paid to a MERCHANT in a round amount (>= Rs 1,000, multiple of 500)
         -> merchant_collusion (a swapped QR sends the money to a person, not the shop)
      5. velocity_bot rows paid to a person whose account gets payments from 2+ other senders
         within 24 h -> mule_account (bots pay gaming / e-commerce merchants)
    """
    raw = load_raw().filter(pl.col("split") == "train").select(
        "txn_id", "ts", "sender_id", "txn_type", "receiver_type", "amount", "sim_age_hours", "device_age_days")
    fl = load_features("flow_v1", "full").select("txn_id", "pair_rev_tiny_n_past_24h", "pair_rev_n_future_24h",
                                                "rcv_in_senders_past_24h", "rcv_in_senders_future_24h")
    d = out.join(raw, on="txn_id", how="left", maintain_order="left").join(fl, on="txn_id", how="left", maintain_order="left")
    real = pl.col("status").is_in(["fraud", "wrong_type", "unreported_typed"])
    lab = pl.col("label_clean")
    sender_refunds = d.group_by("sender_id").agg(_srf=(pl.col("txn_type") == "REFUND").sum())
    d = d.join(sender_refunds, on="sender_id", how="left", maintain_order="left")
    ep = d.filter(real & lab.is_in(["sim_swap", "account_takeover"])).group_by("sender_id", pl.col("ts").dt.date().alias("_day")).agg(
        _sim=(pl.col("sim_age_hours") < 72).any(), _dev=(pl.col("device_age_days") < 3).any())
    d = d.with_columns(_day=pl.col("ts").dt.date()).join(ep, on=["sender_id", "_day"], how="left", maintain_order="left")
    cash_x = ((pl.col("amount") <= 10) & (pl.col("pair_rev_n_future_24h").fill_null(0) > 0)) | (pl.col("pair_rev_tiny_n_past_24h").fill_null(0) > 0)
    new = (
        pl.when(real & (lab == "refund_abuse") & (pl.col("txn_type") != "REFUND") & (pl.col("_srf") == 0)).then(pl.lit("merchant_collusion"))
        .when(real & lab.is_in(["sim_swap", "account_takeover"]) & pl.col("_sim") & ~pl.col("_dev")).then(pl.lit("sim_swap"))
        .when(real & lab.is_in(["sim_swap", "account_takeover"]) & pl.col("_dev") & ~pl.col("_sim")).then(pl.lit("account_takeover"))
        .when(real & (lab == "phishing_collect") & cash_x).then(pl.lit("fake_cashback"))
        .when(real & (lab == "fake_cashback") & (pl.col("txn_type") == "COLLECT") & ~cash_x).then(pl.lit("phishing_collect"))
        .when(real & (lab == "qr_swap") & (pl.col("receiver_type") == "MERCHANT") & (pl.col("amount") >= 1000)
              & ((pl.col("amount") * 100).round(0) % 50000 == 0)).then(pl.lit("merchant_collusion"))
        .when(real & (lab == "velocity_bot") & (pl.col("receiver_type") == "P2P")
              & ((pl.col("rcv_in_senders_past_24h").fill_null(0) + pl.col("rcv_in_senders_future_24h").fill_null(0)) >= 2)).then(pl.lit("mule_account"))
        .otherwise(lab)
    )
    d = d.with_columns(_new=new)
    changed = d["_new"] != d["label_clean"]
    ch = d.filter(changed).group_by("label_clean", "_new").agg(n=pl.len()).sort("n", descending=True)
    print("type rules: rows moved (from -> to)")
    for a, b, n in ch.iter_rows():
        print(f"  {a:<20} -> {b:<20} {n:>6}")
    return d.with_columns(
        label_clean=pl.col("_new"),
        status=pl.when(changed).then(pl.lit("type_rule")).otherwise(pl.col("status")),
    ).select(out.columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--from-changes", action="store_true", help="rebuild from audit/labels_clean_changes.csv")
    ap.add_argument("--type-rules", action="store_true", help="apply the switched-off type rules (experiment only)")
    args = ap.parse_args()
    global TYPE_RULES
    TYPE_RULES = TYPE_RULES or args.type_rules
    if args.from_changes:
        return from_changes()
    probs = crossfit() if args.refit or not PROBS.exists() else pl.read_parquet(PROBS)
    raw = load_raw().filter(pl.col("split") == "train").select("txn_id", "label", "month_idx")
    assert probs["txn_id"].equals(raw["txn_id"])
    P = probs.select(LABELS).to_numpy().astype(np.float64)
    y = raw["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    p_fraud = 1.0 - P[:, 0]
    F = P[:, 1:] / np.maximum(P[:, 1:].sum(axis=1, keepdims=True), 1e-12)   # type, given that it is fraud
    top = F.argmax(axis=1) + 1
    top_share = F.max(axis=1)
    own_share = np.where(y > 0, F[np.arange(len(y)), np.maximum(y, 1) - 1], 0.0)

    fraud_lab = y > 0
    noise = fraud_lab & (p_fraud < NOISE_BELOW)
    ambiguous = fraud_lab & (p_fraud >= NOISE_BELOW) & (p_fraud < REAL_FROM)
    real = fraud_lab & (p_fraud >= REAL_FROM)
    # real fraud whose label is not the type the model sees, by a wide margin
    wrong_type = real & (top != y) & (top_share >= 0.80) & (own_share < 0.05)
    unreported = (~fraud_lab) & (p_fraud >= UNREPORTED_FROM)
    unrep_typed = unreported & (top_share >= 0.80)

    status = np.full(len(y), "legit", dtype=object)
    status[real] = "fraud"
    status[noise] = "noise"
    status[ambiguous] = "ambiguous"
    status[wrong_type] = "wrong_type"
    status[unreported] = "unreported"
    status[unrep_typed] = "unreported_typed"
    label_clean = y.copy()
    label_clean[noise] = 0
    label_clean[wrong_type] = top[wrong_type]
    label_clean[unrep_typed] = top[unrep_typed]
    keep = ~(ambiguous | (unreported & ~unrep_typed))

    out = pl.DataFrame({
        "txn_id": raw["txn_id"], "label_clean": [LABELS[i] for i in label_clean], "keep": keep,
        "status": status.astype(str), "p_fraud": p_fraud.astype(np.float32),
        "top_fraud": [LABELS[i] for i in top], "top_share": top_share.astype(np.float32),
    })
    if TYPE_RULES:
        out = apply_type_rules(out)
        label_clean = out["label_clean"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    out.write_parquet(OUT)
    CHANGES.parent.mkdir(exist_ok=True)
    out.filter(~pl.col("status").is_in(["legit", "fraud"])).select("txn_id", "label_clean", "keep", "status").write_csv(CHANGES)

    n_f = int(fraud_lab.sum())
    print(f"train rows {len(y)}; fraud-labelled {n_f} ({n_f / len(y):.2%})")
    print(f"  real fraud        {int(real.sum()):>7}  ({real.sum() / n_f:.1%} of fraud labels)")
    print(f"  noise -> legit    {int(noise.sum()):>7}  ({noise.sum() / n_f:.1%})")
    print(f"  ambiguous, left out {int(ambiguous.sum()):>5}")
    print(f"  wrong type, relabelled {int(wrong_type.sum()):>5}  ({wrong_type.sum() / max(real.sum(), 1):.1%} of real fraud)")
    print(f"  unreported fraud  {int(unreported.sum()):>7}  (relabelled {int(unrep_typed.sum())}, left out {int((unreported & ~unrep_typed).sum())})")
    print(f"clean fraud share {np.mean((label_clean > 0) & keep):.2%}")
    bins = [0, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0001]
    hf, hl = np.histogram(p_fraud[fraud_lab], bins)[0], np.histogram(p_fraud[~fraud_lab], bins)[0]
    print(pl.DataFrame({"P(fraud)": [f"{bins[i]:.2f}-{min(bins[i + 1], 1):.2f}" for i in range(len(bins) - 1)],
                        "fraud_labelled": hf, "legit_labelled": hl, "fraud_rate": (hf / np.maximum(hf + hl, 1)).round(3)}))
    per = []
    for c in range(1, N_CLASSES):
        m = y == c
        per.append({"class": LABELS[c], "labelled": int(m.sum()), "real": int((m & real).sum()),
                    "noise_share": round(float((m & noise).sum() / m.sum()), 3),
                    "wrong_type_out": int((m & wrong_type).sum()),
                    "clean_total": int(((label_clean == c) & keep).sum())})
    print(pl.DataFrame(per))
    # what the model thinks real fraud rows are, against their label (rows = label, columns = model)
    rf = np.where(real)[0]
    cm = np.zeros((N_CLASSES - 1, N_CLASSES - 1), dtype=np.int64)
    np.add.at(cm, (y[rf] - 1, top[rf] - 1), 1)
    short = [n[:7] for n in LABELS[1:]]
    print("real fraud rows: label (rows) against the model's fraud type (columns)")
    print(pl.DataFrame({"label": LABELS[1:], **{short[j]: cm[:, j] for j in range(N_CLASSES - 1)}}))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
