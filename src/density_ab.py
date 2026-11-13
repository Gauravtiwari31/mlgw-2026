"""Density A/B test (research.md A1, D2; work/L4 task 2).

Question: test holds about half of October's rows, so every count feature is half as large in
test as in a train month. Should training features be built at full density or at half density?

Method: six quick count features, built in the 'full' and 'half_s0' views. LightGBM on Jul+Aug
(same settings as configs/L1_lgbm_base1_s0.yaml), scored on Sep. Arms:

  base only      no count features (the floor)
  full -> half   train rows at full density, Sep rows at half density  (what happens if we ignore A1)
  half -> half   train rows and Sep rows at half density                (the density-matched plan)
  full -> full   train rows and Sep rows at full density                (the number a naive CV shows)

"Sep at half density" copies the test month: each Sep row sees only the Sep rows in its own half.

    python -m src.density_ab                 # 3 seeds, writes audit/density_ab.md
    python -m src.density_ab --seeds 0       # quicker
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, ROOT, load_features, load_folds, load_raw
from .train import fit, train_mask_and_weights
from .windows import attach, build_events, window_agg

AUDIT = ROOT / "audit"
BASE_CONFIG = ROOT / "configs" / "L1_lgbm_base1_s0.yaml"
COUNT_FEATURES = ["rcv_n_1h", "rcv_senders_1h", "rcv_n_24h", "rcv_senders_24h", "dev_n_5m", "snd_n_1h"]
ARMS = [  # name, view of the training rows, view of the Sep rows (None = no count features)
    ("base only", None, None),
    ("full -> half", "full", "half_s0"),
    ("half -> half", "half_s0", "half_s0"),
    ("full -> full", "full", "full"),
]


def count_features(raw: pl.DataFrame, view: str) -> pl.DataFrame:
    """Six counts around each row (both sides of the row, the row itself not counted)."""
    ev = build_events(raw, view, max_window=timedelta(hours=24))
    n = pl.len() - 1
    frames = [
        window_agg(ev, "receiver_id", timedelta(hours=1), "around",
                   [n.alias("rcv_n_1h"), pl.col("sender_id").n_unique().alias("rcv_senders_1h")]),
        window_agg(ev, "receiver_id", timedelta(hours=24), "around",
                   [n.alias("rcv_n_24h"), pl.col("sender_id").n_unique().alias("rcv_senders_24h")]),
        window_agg(ev, "device_id", timedelta(minutes=5), "around", [n.alias("dev_n_5m")]),
        window_agg(ev, "sender_id", timedelta(hours=1), "around", [n.alias("snd_n_1h")]),
    ]
    out = attach(raw, *frames)
    assert out.columns == ["txn_id", *COUNT_FEATURES]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    t0 = time.time()

    raw = load_raw().select("txn_id", "ts", "month_idx", "split", "label", "sender_id", "receiver_id", "device_id")
    meta = raw.select("txn_id", "split", "label", "month_idx").join(
        load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left"
    )
    y = meta["label"].replace_strict(LABEL_TO_ID, default=-1, return_dtype=pl.Int32).to_numpy()
    fold = meta["month_fold"].to_numpy()
    month = meta["month_idx"].to_numpy()
    va = np.where(fold == 2)[0]

    base = load_features("base_v1", "full")
    assert base["txn_id"].equals(meta["txn_id"])
    base_names = [c for c in base.columns if c != "txn_id"]
    Xb = base.select(pl.col(base_names).cast(pl.Float32)).to_numpy()

    counts = {}
    for view in ("full", "half_s0"):
        f = count_features(raw, view)
        assert f["txn_id"].equals(meta["txn_id"])
        counts[view] = f.select(pl.col(COUNT_FEATURES).cast(pl.Float32)).to_numpy()
        print(f"built count features, view={view}  ({time.time() - t0:.0f}s)", flush=True)

    # How far apart are the feature values? (mean per feature: train months vs Sep vs test)
    density = []
    for j, name in enumerate(COUNT_FEATURES):
        density.append({
            "feature": name,
            "jul_aug_full": float(counts["full"][month < 2, j].mean()),
            "jul_aug_half": float(counts["half_s0"][month < 2, j].mean()),
            "sep_half": float(counts["half_s0"][month == 2, j].mean()),
            "test": float(counts["half_s0"][month == 3, j].mean()),
        })

    cfg0 = yaml.safe_load(BASE_CONFIG.read_text())
    results = []
    for seed in args.seeds:
        cfg = {**cfg0, "seed": seed}
        keep, w = train_mask_and_weights(meta, y, cfg)
        tr = np.where(keep & (fold != 2))[0]
        for arm, train_view, val_view in ARMS:
            if train_view is None:
                X, names = Xb, base_names
            else:
                C = counts[train_view].copy()
                C[va] = counts[val_view][va]
                X, names = np.hstack([Xb, C]), base_names + COUNT_FEATURES
            t1 = time.time()
            booster, best = fit(X, y, w, tr, va, cfg, names)
            pred = booster.predict(X[va], num_iteration=best).argmax(axis=1)
            f1 = metric.f1_per_class(y[va], pred)
            results.append({
                "arm": arm, "seed": seed, "best_rounds": int(best), "macro_f1": float(f1.mean()),
                "f1": [float(v) for v in f1],
                "pred_fraud_share": float((pred != 0).mean()),
                "pred_counts": np.bincount(pred, minlength=N_CLASSES).tolist(),
            })
            print(f"seed {seed}  {arm:<13} macro-F1 {f1.mean():.4f}  rounds {best}  ({time.time() - t1:.0f}s)", flush=True)
            del X

    AUDIT.mkdir(exist_ok=True)
    true_counts = np.bincount(y[va], minlength=N_CLASSES).tolist()
    payload = {"seeds": args.seeds, "config": BASE_CONFIG.relative_to(ROOT).as_posix(), "density": density,
               "sep_true_counts": true_counts, "results": results}
    (AUDIT / "density_ab.json").write_text(json.dumps(payload, indent=2))
    write_report(payload, AUDIT / "density_ab.md")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> audit/density_ab.md")


def write_report(p: dict, path: Path) -> None:
    seeds = p["seeds"]
    by = {(r["arm"], r["seed"]): r for r in p["results"]}
    arms = [a for a, _, _ in ARMS]
    mean = {a: float(np.mean([by[a, s]["macro_f1"] for s in seeds])) for a in arms}
    f1_mean = {a: np.mean([by[a, s]["f1"] for s in seeds], axis=0) for a in arms}
    diffs = [by["half -> half", s]["macro_f1"] - by["full -> half", s]["macro_f1"] for s in seeds]
    d_mean, d_min, d_max = float(np.mean(diffs)), min(diffs), max(diffs)
    n_true = np.array(p["sep_true_counts"])
    true_fraud_share = float(n_true[1:].sum() / n_true.sum())

    if d_min > 0 and d_mean >= 0.005:
        verdict = (f"**Verdict: build training features at half density.** Half-density training beats "
                   f"full-density training by {d_mean:+.4f} macro-F1 on half-density Sep, and wins on every seed.")
    elif d_max < 0 and d_mean <= -0.005:
        verdict = (f"**Verdict: full-density training is better.** Half-density training is {d_mean:+.4f} "
                   f"macro-F1 against full-density training on half-density Sep, and loses on every seed.")
    else:
        verdict = (f"**Verdict: no clear difference.** Half minus full is {d_mean:+.4f} macro-F1 "
                   f"(per seed {d_min:+.4f} to {d_max:+.4f}), which is inside the noise.")

    L = []
    L.append("# Density A/B test\n")
    L.append("Generated by `python -m src.density_ab`. Do not edit by hand; rerun instead.\n")
    L.append(verdict + "\n")
    L.append("## Setup\n")
    L.append(f"- Features: `base_v1` plus six counts ({', '.join('`' + c + '`' for c in COUNT_FEATURES)}), "
             "each counting other rows on both sides of the row.")
    L.append(f"- Model: LightGBM, settings from `{p['config']}`, train Jul+Aug, early stopping on Sep macro-F1, plain argmax.")
    L.append(f"- Seeds: {', '.join(str(s) for s in seeds)} (the seed changes the legit downsampling and the tree sampling).")
    L.append("- \"Sep at half density\" means each Sep row sees only the Sep rows in its own `half_s0` half, as in the test month.\n")
    L.append("## Macro-F1 on Sep\n")
    L.append("| Arm | Train rows | Sep rows | Mean | " + " | ".join(f"Seed {s}" for s in seeds) + " | Rounds | Predicted fraud share |")
    L.append("|---|---|---|---|" + "---|" * len(seeds) + "---|---|")
    views = {a: (tv, vv) for a, tv, vv in ARMS}
    for a in arms:
        tv, vv = views[a]
        per_seed = " | ".join(f"{by[a, s]['macro_f1']:.4f}" for s in seeds)
        rounds = "/".join(str(by[a, s]["best_rounds"]) for s in seeds)
        share = np.mean([by[a, s]["pred_fraud_share"] for s in seeds])
        L.append(f"| {a} | {tv or '-'} | {vv or '-'} | **{mean[a]:.4f}** | {per_seed} | {rounds} | {share:.2%} |")
    L.append(f"\nTrue fraud share in Sep labels: {true_fraud_share:.2%}.")
    L.append(f"Half minus full, both scored on half-density Sep: mean {d_mean:+.4f}, per seed "
             + ", ".join(f"{d:+.4f}" for d in diffs) + ".")
    L.append(f"Count features over base only (half -> half): {mean['half -> half'] - mean['base only']:+.4f}.")
    L.append(f"What a naive full-density CV would have reported: {mean['full -> full']:.4f}, "
             f"which is {mean['full -> full'] - mean['full -> half']:+.4f} above what the same training gets at test density.\n")
    L.append("## Per-class F1 on Sep (mean over seeds)\n")
    L.append("| Class | Sep rows | " + " | ".join(arms) + " | half minus full |")
    L.append("|---|---|" + "---|" * len(arms) + "---|")
    for c in range(N_CLASSES):
        vals = " | ".join(f"{f1_mean[a][c]:.3f}" for a in arms)
        L.append(f"| {LABELS[c]} | {n_true[c]} | {vals} | {f1_mean['half -> half'][c] - f1_mean['full -> half'][c]:+.3f} |")
    L.append("\n## Mean value of each count feature\n")
    L.append("| Feature | Jul+Aug, full view | Jul+Aug, half view | Sep, half view | Test month |")
    L.append("|---|---|---|---|---|")
    for d in p["density"]:
        L.append(f"| `{d['feature']}` | {d['jul_aug_full']:.2f} | {d['jul_aug_half']:.2f} | {d['sep_half']:.2f} | {d['test']:.2f} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
