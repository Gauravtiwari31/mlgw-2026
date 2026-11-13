"""Config-driven MLP trainer (L3, work/L3 TASKS.md #8, model M6).

    python -m src.train_nn --config configs/L3_nn_base1-flow1-beh1_s0.yaml             # full run
    python -m src.train_nn --config configs/L3_nn_base1-flow1-beh1_s0.yaml --mode sep # honest score only

Same contracts as src.train: the same load_matrix (features, view, labels, drop_features), the same
row weights and legit downsampling, the same month-fold CV, the same outputs (oof/, preds/,
runs/<run_id>.json, key model=mlp).

Design (TASKS.md #8): standardised numeric features, one embedding per cat_* column, weighted
cross-entropy, month-fold OOF. The scaler and the per-column means/vars are fit on each fold's
TRAINING rows only; NaNs are filled with 0 after scaling. Early stopping monitors the plain
macro-F1 on the validation fold. GPU training is not bit-deterministic; the seed fixes everything
else. Keep runs on one GPU comparable by seed, not by exact reproducibility.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import yaml

from . import metric
from .common import N_CLASSES, OOF, PREDS, RUNS
from .train import _git_commit, _save_probs, load_matrix, train_mask_and_weights


class MLP(nn.Module):
    def __init__(self, n_num: int, cats: list[int], cfg: dict):
        super().__init__()
        emb_dim = lambda card: max(2, min(16, round(math.sqrt(card))))
        self.embs = nn.ModuleList([nn.Embedding(card, emb_dim(card)) for card in cats])
        d_in = n_num + sum(emb_dim(c) for c in cats)
        h = cfg.get("hidden", [512, 256])
        p = cfg.get("dropout", 0.2)
        layers: list[nn.Module] = []
        last = d_in
        for width in h:
            layers += [nn.Linear(last, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(p)]
            last = width
        layers += [nn.Linear(last, N_CLASSES)]
        self.body = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        parts = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return self.body(torch.cat([x_num, *parts], dim=1))


def _split_cat_num(X: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """cat_* columns -> int32 index array; the rest -> float32 array."""
    cat_cols = [c for c in names if c.startswith("cat_")]
    num_cols = [c for c in names if not c.startswith("cat_")]
    xc = X[cat_cols].to_numpy(dtype=np.int32, copy=True)
    cats = [int(X[c].max()) + 1 for c in cat_cols]  # dense codes over pooled train + test
    xn = X[num_cols].to_numpy(dtype=np.float32, copy=True)
    return xn, xc, cats


class _Scaler:
    """Standardise with train-fold statistics; NaN -> 0 (the mean) after scaling."""

    def fit(self, x: np.ndarray) -> "_Scaler":
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)


def fit(X_num, X_cat, y, w, tr_idx, va_idx, cfg, cats, rounds=None):
    """Train one MLP. rounds=None: early stopping on the validation fold's macro-F1."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = _Scaler().fit(X_num[tr_idx])
    xn_t = torch.tensor(sc.transform(X_num[tr_idx]), device=device)
    xn_v = torch.tensor(sc.transform(X_num[va_idx]), device=device)
    xc_t = torch.tensor(X_cat[tr_idx], device=device)
    xc_v = torch.tensor(X_cat[va_idx], device=device)
    y_t = torch.tensor(y[tr_idx], device=device, dtype=torch.long)
    w_t = torch.tensor(w[tr_idx], device=device, dtype=torch.float32)
    y_v = torch.tensor(y[va_idx], device=device, dtype=torch.long)

    torch.manual_seed(cfg.get("seed", 0))
    model = MLP(xn_t.shape[1], cats, cfg.get("nn", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    bs = cfg.get("batch", 4096)
    n = xn_t.shape[0]
    steps = max(1, n // bs)  # drop the tail batch: BatchNorm cannot train on a size-1 batch
    max_epochs = cfg.get("epochs", 40) if rounds is None else int(rounds)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.get("lr", 1e-3), total_steps=steps * max_epochs, pct_start=0.1
    ) if rounds is None else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps)
    lossf = nn.CrossEntropyLoss(reduction="none")

    best_f1, best_state, best_ep = -1.0, None, 1
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            logits = model(xn_t[idx], xc_t[idx])
            loss = (lossf(logits, y_t[idx]) * w_t[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(xn_v, xc_v).argmax(dim=1).cpu().numpy()
        f1 = metric.macro_f1(y[va_idx], pv)
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            if rounds is None:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= cfg.get("patience", 5) and rounds is None:
            break
    if rounds is None and best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        probs = model(xn_v, xc_v).softmax(dim=1).cpu().numpy()
    return model, sc, probs, best_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["all", "sep"], default="all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = cfg["run_id"]
    t0 = time.time()

    meta, X, y, names = load_matrix(cfg)
    keep, w = train_mask_and_weights(meta, y, cfg)
    fold = meta["month_fold"].to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    X = pd.DataFrame(X, columns=names)
    for n_ in names:
        if n_.startswith("cat_"):
            X[n_] = X[n_].astype("int32")
    X_num, X_cat, cats = _split_cat_num(X, names)
    print(f"run {run_id}: {len(names)} features ({len(cats)} categorical), "
          f"{int(keep.sum())} training rows after sampling, "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    va = np.where(fold == 2)[0]
    tr = np.where(keep & (fold != 2))[0]
    model, sc, pv, best_ep = fit(X_num, X_cat, y, w, tr, va, cfg, cats)
    oof = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    oof[va] = pv
    rep = metric.report(y[va], oof[va].argmax(axis=1),
                        f"\nSep validation, {best_ep} epochs (plain argmax)")

    log = {
        "run_id": run_id, "config": args.config, "commit": _git_commit(), "model": "mlp",
        "features": cfg["features"], "view": cfg.get("view", "half_s0"),
        "seed": cfg.get("seed", 0), "best_rounds": int(best_ep),
        "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"], "n_features": len(names),
    }
    if args.mode == "all":
        for k in (0, 1):
            va_k = np.where(fold == k)[0]
            tr_k = np.where(keep & (fold != k))[0]
            _, _, pvk, _ = fit(X_num, X_cat, y, w, tr_k, va_k, cfg, cats, rounds=best_ep)
            oof[va_k] = pvk
            print(f"fold {k} OOF macro-F1 = {metric.macro_f1(y[va_k], oof[va_k].argmax(axis=1)):.4f}")
        cv_rows = np.where(fold >= 0)[0]
        _save_probs(OOF / f"{run_id}.parquet", meta["txn_id"][cv_rows], oof[cv_rows])
        # full fit on all months: 15% more epochs, mirroring src.train's round scaling.
        # A tiny dummy validation index keeps the fold tensors off the 4 GB card.
        model_full, sc_full, _, _ = fit(
            X_num, X_cat, y, w, np.where(keep)[0], va[:2048], cfg, cats,
            rounds=int(best_ep * 1.15) + 1,
        )
        te = np.where(is_test)[0]
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            xnt = torch.tensor(sc_full.transform(X_num[te]), device=device)
            xct = torch.tensor(X_cat[te], device=device)
            probs = model_full(xnt, xct).softmax(dim=1).cpu().numpy()
        _save_probs(PREDS / f"{run_id}.parquet", meta["txn_id"][te], probs)
        log["oof_macro_f1"] = metric.macro_f1(y[cv_rows], oof[cv_rows].argmax(axis=1))
        print(f"all-months OOF macro-F1 = {log['oof_macro_f1']:.4f}")

    log["minutes"] = round((time.time() - t0) / 60, 1)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps(log, indent=2))
    print(f"done in {log['minutes']} min -> runs/{run_id}.json")


if __name__ == "__main__":
    main()
