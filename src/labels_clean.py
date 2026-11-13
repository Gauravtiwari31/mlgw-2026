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

from .common import CACHE, LABEL_TO_ID, LABELS, N_CLASSES, ROOT, load_features, load_raw

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--from-changes", action="store_true", help="rebuild from audit/labels_clean_changes.csv")
    args = ap.parse_args()
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
