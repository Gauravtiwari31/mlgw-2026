"""Pairwise sim_swap vs account_takeover disambiguation on top of a mean blend.

Motivation (audit/leaderboard_analysis.md + audit/strategy_day2.md, item 2): sim_swap and
account_takeover are each other's main confusion in every model (about 30% of their train labels
carry the other type's signature). A dedicated binary model sees only those two classes and can
separate them better than the 12-class head, whose gradient is dominated by the legit boundary.

Method, applied to a mean blend of run OOFs / preds:
  1. mean the runs' probabilities (the blend),
  2. train CatBoost binary (CPU, seconds: the two classes have ~10k train rows) on clean labels,
     same features and view as the blend runs, early stopped on Sep,
  3. for every row whose blend argmax is sim_swap or account_takeover, rescale the pair's
     probabilities to the binary model's odds:  p_sim = q * (p_sim + p_ato), p_ato = (1-q) * (...).
     The total fraud mass of the pair and every other class stay untouched.
  4. report Sep per-class F1 before/after (clean labels, keep rows only). Writes a variant
     submission only when the caller passes --write.

    python -m src.simato_hybrid --runs RUN1 RUN2 ... [--write]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from . import metric
from .common import LABELS, ROOT, SUBS
from .postprocess import write_submission
from .train import load_matrix, train_mask_and_weights

SIM, ATO = LABELS.index("sim_swap"), LABELS.index("account_takeover")


def load_probs(paths: list[Path]) -> tuple[pl.DataFrame, np.ndarray]:
    """Mean of run oof/preds: returns the txn_id frame and (n, 12) mean probability matrix."""
    oofs, preds = [], []
    ids = None
    for run in paths:
        oofs.append(pl.read_parquet(ROOT / "oof" / f"{run}.parquet"))
        preds.append(pl.read_parquet(ROOT / "preds" / f"{run}.parquet"))
    ids = oofs[0]["txn_id"]
    O = np.mean([o.select(LABELS).to_numpy() for o in oofs], axis=0)
    P = np.mean([p.select(LABELS).to_numpy() for p in preds], axis=0)
    # oof frames cover only train rows; align them onto the full id list via a join-free concat
    return ids, O, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--write", default=None, help="name of the variant submission to write")
    args = ap.parse_args()

    cfg = {
        "features": ["base_v1", "flow_v1", "behaviour_v1", "behaviour_v2", "behaviour_v3",
                     "labelhist_v2"],
        "view": "fw_s0", "labels": "clean", "drop_features": ["ip_risk_rank"],
        "seed": 0, "class_weight_power": 0.0, "legit_keep": 1.0,
        "num_boost_round": 2000, "early_stopping": 100,
        "params": {"learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0, "verbose": 100},
    }

    meta, X, y, names = load_matrix(cfg)
    keep, _ = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    ok = meta["keep"].fill_null(True).to_numpy() if "keep" in meta.columns else np.ones(meta.height, dtype=bool)
    is_test = (meta["split"] == "test").to_numpy()

    # blend matrices: the oof parquet covers exactly the cv rows in file order
    ids, O, P = load_probs([Path(r) for r in args.runs])
    cv_rows = np.where(fold >= 0)[0]
    assert len(ids) == len(cv_rows), f"oof rows {len(ids)} != cv rows {len(cv_rows)}"
    id_list = ids.to_list()
    txn_list = meta["txn_id"].to_list()
    assert id_list[0] == txn_list[cv_rows[0]] and id_list[-1] == txn_list[cv_rows[-1]]
    O_full = np.zeros((meta.height, 12), dtype=np.float32)
    P_full = np.zeros((meta.height, 12), dtype=np.float32)
    O_full[cv_rows] = O
    te_rows = np.where(is_test)[0]
    assert P.shape[0] == len(te_rows), "preds must be exactly the test rows"
    P_full[te_rows] = P

    from catboost import CatBoostClassifier, Pool

    # binary target: sim_swap = 1, account_takeover = 0, on clean-labelled train rows only
    bi = np.where(keep & ok & ((y == SIM) | (y == ATO)))[0]
    yb = (y[bi] == SIM).astype(np.int32)
    va_mask = fold[bi] == 2
    tr_mask = ~va_mask
    model = CatBoostClassifier(**cfg["params"], iterations=cfg["num_boost_round"], random_seed=0)
    model.fit(Pool(X[bi[tr_mask]], yb[tr_mask]),
              eval_set=Pool(X[bi[va_mask]], yb[va_mask]), use_best_model=True,
              early_stopping_rounds=cfg["early_stopping"])
    print(f"binary sim-vs-ATO: {int(tr_mask.sum())} train rows, "
          f"best_iter={model.get_best_iteration()}")

    def retype(Pr: np.ndarray) -> np.ndarray:
        am = Pr.argmax(axis=1)
        sel = (am == SIM) | (am == ATO)
        out = Pr.copy()
        q = model.predict_proba(X[sel])[:, 1].astype(np.float32)
        pair = Pr[sel, SIM] + Pr[sel, ATO]
        out[sel, SIM] = q * pair
        out[sel, ATO] = (1.0 - q) * pair
        return out

    va_sep = np.where((fold == 2) & ok)[0]
    before = metric.report(y[va_sep], O_full[va_sep].argmax(axis=1), "\nBlend, Sep before retype")
    after_pr = retype(O_full)
    after = metric.report(y[va_sep], after_pr[va_sep].argmax(axis=1), "\nBlend, Sep after retype")
    print(f"sim_swap F1 {before['f1']['sim_swap']:.4f} -> {after['f1']['sim_swap']:.4f}, "
          f"account_takeover F1 {before['f1']['account_takeover']:.4f} -> {after['f1']['account_takeover']:.4f}, "
          f"macro {before['macro_f1']:.4f} -> {after['macro_f1']:.4f}")

    if args.write and after["macro_f1"] > before["macro_f1"] + 0.002:
        # retype the sim/ATO calls of an existing tuned submission; other labels untouched
        sub = pl.read_csv(SUBS / f"{args.write}.csv")
        label_of = dict(zip(sub["id"].to_list(), sub["label"].to_list()))
        te_labels = np.array([label_of[t] for t in meta["txn_id"][te_rows].to_list()])
        sel = np.where(np.isin(te_labels, ["sim_swap", "account_takeover"]))[0]
        q = model.predict_proba(X[te_rows[sel]])[:, 1]
        conf = np.where(te_labels[sel] == "sim_swap", q, 1.0 - q)
        flip = conf < 0.5
        new_labels = te_labels.copy()
        new_labels[sel[flip]] = np.where(q[flip] > 0.5, "sim_swap", "account_takeover")
        n_flip = int(flip.sum())
        print(f"retype: {len(sel)} sim/ATO calls in the submission, {n_flip} flipped "
              f"(mean confidence of flips {conf[flip].mean() if n_flip else 0:.3f})")
        out = pl.DataFrame({"id": meta["txn_id"][te_rows], "label": new_labels})
        write_submission(out, args.write + "_simato", {"base": args.write, "flips": n_flip})
    elif args.write:
        print("no gain above +0.002 on Sep; variant not written")


if __name__ == "__main__":
    main()
