"""Average several runs, smooth fraud types inside an episode, and write a submission.
Everything is fitted and scored against CLEAN labels (labels_clean.parquet).

Steps, each checked on Sep and kept only if it is switched on:
  1. mean of the runs' probabilities (seeds, views)
  2. --smooth: rows the blend calls fraud are grouped into episodes (same sender, gaps under
     60 min; then same personal receiver, gaps under 24 h). Inside an episode the fraud TYPE
     distribution is averaged, so one episode gets one type. P(legit) of each row is untouched.
  3. --weights: class weights for the argmax, searched in a narrow range (0.5x to 2x) on Jul+Aug.
     The wide search in src.postprocess picked extreme values (0.05) that mean nothing.

    python -m src.blend --name fwclean_v1 --runs RUN1 RUN2 ... --smooth --weights
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT, load_folds, load_raw
from .postprocess import write_submission

GRID = np.exp(np.linspace(-0.7, 0.7, 15))


def fit_weights(P: np.ndarray, y: np.ndarray, n_pass: int = 2) -> np.ndarray:
    w = np.ones(N_CLASSES)
    best = metric.macro_f1(y, (P * w).argmax(axis=1))
    for _ in range(n_pass):
        for c in range(1, N_CLASSES):
            for g in GRID:
                t = w.copy()
                t[c] = g
                s = metric.macro_f1(y, (P * t).argmax(axis=1))
                if s > best + 2e-4:   # ignore gains too small to be real
                    best, w = s, t
    return w


def _episodes(keys: np.ndarray, ts: np.ndarray, gap: float) -> np.ndarray:
    """Group ids for rows sorted by (key, ts): a new group when the key changes or the gap is exceeded."""
    order = np.lexsort((ts, keys))
    k, t = keys[order], ts[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (k[1:] != k[:-1]) | ((t[1:] - t[:-1]) > gap)
    gid = np.empty(len(order), dtype=np.int64)
    gid[order] = np.cumsum(new) - 1
    return gid


def smooth(P: np.ndarray, info: pl.DataFrame) -> np.ndarray:
    """Average the fraud-type distribution inside each episode of rows called fraud."""
    P = P.copy()
    ts = info["tsec"].to_numpy()
    for key, gap, mask_col in (("snd", 3600.0, None), ("rcv", 86400.0, "is_p2p")):
        fraud = P.argmax(axis=1) != 0
        if mask_col is not None:
            fraud &= info[mask_col].to_numpy()
        idx = np.where(fraud)[0]
        if len(idx) == 0:
            continue
        gid = _episodes(info[key].to_numpy()[idx], ts[idx], gap)
        F = P[idx, 1:]
        tot = F.sum(axis=1, keepdims=True)
        F = F / np.maximum(tot, 1e-12)
        n_g = gid.max() + 1
        S = np.zeros((n_g, N_CLASSES - 1))
        np.add.at(S, gid, F)
        cnt = np.bincount(gid, minlength=n_g)[:, None]
        P[idx, 1:] = (S / cnt)[gid] * tot
    return P


def _info(ids: pl.Series) -> pl.DataFrame:
    raw = load_raw().select(
        "txn_id", snd=pl.col("sender_id").str.slice(2).cast(pl.Int64),
        rcv=pl.col("receiver_id").str.slice(2).cast(pl.Int64) + pl.when(pl.col("receiver_type") == "P2P").then(0).otherwise(10**9),
        is_p2p=pl.col("receiver_type") == "P2P", tsec=pl.col("ts").dt.epoch("s").cast(pl.Float64),
    )
    return pl.DataFrame({"txn_id": ids}).join(raw, on="txn_id", how="left", maintain_order="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()

    oofs = [pl.read_parquet(OOF / f"{r}.parquet") for r in args.runs]
    tests = [pl.read_parquet(PREDS / f"{r}.parquet") for r in args.runs]
    for o, t in zip(oofs, tests):
        assert o["txn_id"].equals(oofs[0]["txn_id"]) and t["txn_id"].equals(tests[0]["txn_id"])
    ids = oofs[0]["txn_id"]
    truth = load_raw().select("txn_id", "label").join(load_folds().select("txn_id", "month_fold"), on="txn_id")
    cl = pl.read_parquet(ROOT / "labels_clean.parquet").select("txn_id", "label_clean", "keep")
    j = pl.DataFrame({"txn_id": ids}).join(truth, on="txn_id", how="left", maintain_order="left").join(
        cl, on="txn_id", how="left", maintain_order="left")
    y = j["label_clean"].fill_null(j["label"]).replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    y_raw = j["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    keep = j["keep"].fill_null(True).to_numpy()
    fold = j["month_fold"].to_numpy()
    sep, rest = (fold == 2) & keep, (fold != 2) & keep

    print("single runs, Sep clean macro-F1 (plain argmax):")
    for r, o in zip(args.runs, oofs):
        p = o.select(LABELS).to_numpy()
        print(f"  {r:<44}{metric.macro_f1(y[sep], p[sep].argmax(axis=1)):.4f}")
    P = np.mean([o.select(LABELS).to_numpy().astype(np.float64) for o in oofs], axis=0)
    Pt = np.mean([t.select(LABELS).to_numpy().astype(np.float64) for t in tests], axis=0)
    steps = {"mean of runs": metric.macro_f1(y[sep], P[sep].argmax(axis=1))}

    if args.smooth:
        P = smooth(P, _info(ids))           # episodes never cross months in practice (gaps are short)
        Pt = smooth(Pt, _info(tests[0]["txn_id"]))
        steps["+ episode smoothing"] = metric.macro_f1(y[sep], P[sep].argmax(axis=1))
    w = np.ones(N_CLASSES)
    if args.weights:
        w_fit = fit_weights(P[rest], y[rest])
        steps["+ weights (fitted Jul+Aug)"] = metric.macro_f1(y[sep], (P[sep] * w_fit).argmax(axis=1))
        w = fit_weights(P[keep], y[keep])
        print("weights fitted on Jul-Sep:", {LABELS[c]: round(float(w[c]), 2) for c in range(N_CLASSES) if abs(w[c] - 1) > 1e-9})
    for k, v in steps.items():
        print(f"{k:<30}{v:.4f}")
    w_sep = w_fit if args.weights else w
    rep = metric.report(y[sep], (P[sep] * w_sep).argmax(axis=1), "\nSep against clean labels, final decision rule")
    all_sep = fold == 2
    print(f"same decisions against RAW Sep labels: {metric.macro_f1(y_raw[all_sep], (P[all_sep] * w_sep).argmax(axis=1)):.4f}")

    lab = (Pt * w).argmax(axis=1)
    cnt_t = np.bincount(lab, minlength=N_CLASSES) / len(lab) * 1e4
    cnt_s = np.bincount((P[fold == 2] * w).argmax(axis=1), minlength=N_CLASSES) / (fold == 2).sum() * 1e4
    print("\npredicted rows per 10,000: Sep (out of fold) against the test month")
    for c in range(1, N_CLASSES):
        print(f"  {LABELS[c]:<20}{cnt_s[c]:>8.1f}{cnt_t[c]:>8.1f}{cnt_t[c] / max(cnt_s[c], 1e-9):>7.2f}x")
    conf_s = np.sort(P[fold == 2].max(axis=1)[(P[fold == 2].argmax(axis=1) != 0)])
    conf_t = np.sort(Pt.max(axis=1)[Pt.argmax(axis=1) != 0])
    print(f"confidence of fraud calls (median top probability): Sep {np.median(conf_s):.3f}, test {np.median(conf_t):.3f}; "
          f"share under 0.6: Sep {(conf_s < 0.6).mean():.3f}, test {(conf_t < 0.6).mean():.3f}")
    if args.no_sub:
        return
    sub = pl.DataFrame({"id": tests[0]["txn_id"], "label": [LABELS[i] for i in lab]})
    write_submission(sub, args.name, {
        "run": args.name, "runs": args.runs, "labels": "clean", "smooth": args.smooth,
        "weights": dict(zip(LABELS, map(float, w))), "sep_macro_f1_honest": rep["macro_f1"],
        "sep_steps": steps, "predicted_fraud_share": float((lab != 0).mean()),
    })


if __name__ == "__main__":
    main()
