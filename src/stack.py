"""Stacker: combine the OOF probabilities of several runs into one set of probabilities.

    python -m src.stack --name v1 --runs L2_lgbm_flow1_s0 L3_cat_beh1_s0 ... [--model lr|lgbm|mean]

Inputs are oof/<run>.parquet and preds/<run>.parquet written by src.train (or any script that
follows the same contract). Every input must cover the same rows.

Month-fold CV, as in research.md D2: the stacker is fitted on two months of OOF and predicts
the third, so its Sep score is honest (the Sep OOF of each input came from models trained on
Jul+Aug). It is then refitted on all three months and applied to the test predictions.

Models:
  lr    multinomial logistic regression on the log-probabilities (default)
  lgbm  shallow LightGBM on the log-probabilities
  mean  plain average of the probabilities, nothing fitted (the baseline every stacker must beat)

Output follows the run contract: oof/stack_<name>.parquet, preds/stack_<name>.parquet and
runs/stack_<name>.json. Then: python -m src.postprocess --run stack_<name>
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, RUNS, load_folds, load_raw
from .train import _git_commit, _save_probs

LEGIT_KEEP = 0.25  # legit rows kept when fitting the stacker (weight scaled back up)


def _load(runs: list[str], folder) -> tuple[pl.Series, np.ndarray]:
    ids, blocks = None, []
    for r in runs:
        df = pl.read_parquet(folder / f"{r}.parquet")
        if ids is None:
            ids = df["txn_id"]
        else:
            assert df.height == ids.len(), f"{r}: {df.height} rows, expected {ids.len()}"
            df = pl.DataFrame({"txn_id": ids}).join(df, on="txn_id", how="left", maintain_order="left")
            assert df[LABELS[0]].null_count() == 0, f"{r}: rows missing compared with {runs[0]}"
        blocks.append(df.select(LABELS).to_numpy().astype(np.float64))
    return ids, np.stack(blocks, axis=1)  # rows x runs x classes


def _features(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, 1e-9, 1.0)).reshape(P.shape[0], -1)


def _sample(y: np.ndarray, idx: np.ndarray, salt: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(salt)
    keep = (y[idx] != 0) | (rng.random(len(idx)) < LEGIT_KEEP)
    sel = idx[keep]
    w = np.where(y[sel] == 0, 1.0 / LEGIT_KEEP, 1.0)
    return sel, w


class _LR:
    def fit(self, X, y, w, Xva=None, yva=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X)
        self.m = LogisticRegression(C=1.0, max_iter=300, tol=1e-5)
        self.m.fit(self.scaler.transform(X), y, sample_weight=w)
        assert list(self.m.classes_) == list(range(N_CLASSES)), "a class is missing from the stacker data"
        return self

    def predict(self, X):
        return self.m.predict_proba(self.scaler.transform(X))


class _LGBM:
    params = {"objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
              "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "lambda_l2": 5.0, "verbose": -1, "seed": 0}

    def __init__(self, rounds=None):
        self.rounds = rounds

    def fit(self, X, y, w, Xva=None, yva=None):
        import lightgbm as lgb

        dtr = lgb.Dataset(X, y, weight=w)
        if self.rounds is None:
            dva = lgb.Dataset(Xva, yva, reference=dtr)
            self.m = lgb.train(self.params, dtr, 2000, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
            self.rounds = self.m.best_iteration
        else:
            self.m = lgb.train(self.params, dtr, self.rounds)
        return self

    def predict(self, X):
        return self.m.predict(X, num_iteration=self.rounds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", choices=["lr", "lgbm", "mean"], default="lr")
    args = ap.parse_args()
    run_id = f"stack_{args.name}"
    t0 = time.time()

    ids, P = _load(args.runs, OOF)
    tids, Pt = _load(args.runs, PREDS)
    truth = load_raw().select("txn_id", "label").join(load_folds().select("txn_id", "month_fold"), on="txn_id")
    meta = pl.DataFrame({"txn_id": ids}).join(truth, on="txn_id", how="left", maintain_order="left")
    y = meta["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    fold = meta["month_fold"].to_numpy()
    sep = fold == 2

    print(f"{run_id}: {len(args.runs)} inputs, {P.shape[0]} OOF rows, model={args.model}")
    inputs = {}
    for j, r in enumerate(args.runs):
        inputs[r] = metric.macro_f1(y[sep], P[sep, j].argmax(axis=1))
        print(f"  input {r:<40} Sep macro-F1 (plain argmax) {inputs[r]:.4f}")

    if args.model == "mean":
        oof, test = P.mean(axis=1), Pt.mean(axis=1)
    else:
        X, Xt = _features(P), _features(Pt)
        oof = np.zeros((P.shape[0], N_CLASSES))
        rounds = None
        for k in (2, 0, 1):  # Sep first, so LightGBM's round count comes from the honest fold
            tr, w = _sample(y, np.where(fold != k)[0], salt=k)
            va = fold == k
            model = _LR() if args.model == "lr" else _LGBM(rounds)
            model.fit(X[tr], y[tr], w, X[va], y[va])
            rounds = getattr(model, "rounds", None)
            oof[va] = model.predict(X[va])
            print(f"  fold {k}: macro-F1 {metric.macro_f1(y[va], oof[va].argmax(axis=1)):.4f}")
        tr, w = _sample(y, np.arange(len(y)), salt=9)
        model = _LR() if args.model == "lr" else _LGBM(rounds)
        test = model.fit(X[tr], y[tr], w).predict(Xt)

    rep = metric.report(y[sep], oof[sep].argmax(axis=1), f"\n{run_id}: Sep (plain argmax)")
    best_input = max(inputs.values())
    print(f"stack minus best input on Sep (plain argmax): {rep['macro_f1'] - best_input:+.4f}")
    print("Plain argmax is not a fair comparison: the inputs were trained with class weights, the stacker\n"
          f"was not. Compare after post-processing: python -m src.postprocess --run {run_id} --no-sub")

    _save_probs(OOF / f"{run_id}.parquet", ids, oof.astype(np.float32))
    _save_probs(PREDS / f"{run_id}.parquet", tids, test.astype(np.float32))
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "model": args.model, "inputs": args.runs, "commit": _git_commit(),
        "input_sep_macro_f1": inputs, "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"],
        "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
