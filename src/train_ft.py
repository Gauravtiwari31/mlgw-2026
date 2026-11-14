"""FT-Transformer trainer (L3, torch): each feature is a token, attention over features.

    python -m src.train_ft --config configs/L3_ft_fw-clean_b1-f1-beh12-lh2_s0.yaml --mode sep
    python -m src.train_ft --config configs/L3_ft_fw-clean_b1-f1-beh12-lh2_s0.yaml

Same contracts as src.train_nn: load_matrix, row weights, keep-row scoring, month-fold CV,
oof/ + preds/ + runs/<run_id>.json. The model follows Gorishniy et al. 2021 (FT-Transformer):
per-feature linear tokens for numeric columns, embeddings for cat_* columns, one [CLS] token,
pre-LN transformer blocks with torch 2.x memory-efficient attention (needed on the 4 GB 3050).

Training set: the same weight-corrected legit_keep sample as every other L3 run (0.2). Inference
covers every train and test row (full OOF + full test predictions); the weights rescale the
loss so the sample stays unbiased.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from . import metric
from .common import N_CLASSES, OOF, PREDS, RUNS
from .train import _git_commit, _save_probs, load_matrix, train_mask_and_weights
from .train_nn import _Scaler, _split_cat_num


class Block(nn.Module):
    """Pre-LN transformer block on scaled_dot_product_attention (memory-efficient on 4 GB cards)."""

    def __init__(self, d: int, heads: int, ff_mult: int, p: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(p),
            nn.Linear(ff_mult * d, d), nn.Dropout(p),
        )
        self.heads, self.d, self.p = heads, d, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(b, s, 3, self.heads, self.d // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], dropout_p=self.p if self.training else 0.0
        )
        att = att.transpose(1, 2).reshape(b, s, self.d)
        x = x + self.proj(att)
        return x + self.mlp(self.ln2(x))


class FTTransformer(nn.Module):
    def __init__(self, n_num: int, cats: list[int], cfg: dict):
        super().__init__()
        d = cfg.get("dim", 48)
        p = cfg.get("dropout", 0.2)
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d) for card in cats])
        # per-feature linear tokens as one batched tensor (n_num, d): x_i * w_i + b_i
        self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(d, cfg.get("heads", 8), cfg.get("ff", 4), p)
             for _ in range(cfg.get("layers", 3))]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, N_CLASSES))
        self.d = d

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        b = x_num.shape[0]
        toks = [self.cls.expand(b, -1, -1),
                x_num.unsqueeze(-1) * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)]
        for i, emb in enumerate(self.cat_embs):
            toks.append(emb(x_cat[:, i]).unsqueeze(1))
        x = torch.cat(toks, dim=1)  # (batch, 1 + n_num + n_cat, d)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x[:, 0]))


def fit(X_num, X_cat, y, w, tr_idx, va_idx, cfg, cats, rounds=None):
    """Train one model. rounds=None: early stopping on the validation fold's macro-F1."""
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
    model = FTTransformer(xn_t.shape[1], cats, cfg.get("ft", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("weight_decay", 1e-5))
    bs = cfg.get("batch", 256)
    n = xn_t.shape[0]
    steps = max(1, n // bs)  # drop the tail batch (BatchNorm-free, but keeps step counts clean)
    max_epochs = cfg.get("epochs", 10) if rounds is None else int(rounds)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([
                model(xn_v[s:s + 2048], xc_v[s:s + 2048]).argmax(dim=1).cpu().numpy()
                for s in range(0, xn_v.shape[0], 2048)
            ])
        f1 = metric.macro_f1(y[va_idx], pv)
        print(f"epoch {ep}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            if rounds is None:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= cfg.get("patience", 3) and rounds is None:
            break
    if rounds is None and best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        probs = np.concatenate([
            model(xn_v[s:s + 2048], xc_v[s:s + 2048]).softmax(dim=1).cpu().numpy()
            for s in range(0, xn_v.shape[0], 2048)
        ])
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
    n_tok = 1 + X_num.shape[1] + len(cats)
    print(f"run {run_id}: {len(names)} features ({len(cats)} cat), {n_tok} tokens, "
          f"{int(keep.sum())} training rows, "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    ok = meta["keep"].fill_null(True).to_numpy() if "keep" in meta.columns else np.ones(meta.height, dtype=bool)
    sep_all = np.where(fold == 2)[0]
    va = np.where((fold == 2) & ok)[0]
    tr = np.where(keep & (fold != 2))[0]
    model, sc, pv, best_ep = fit(X_num, X_cat, y, w, tr, va, cfg, cats)
    oof = np.zeros((meta.height, N_CLASSES), dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model.eval()
        # Sep probabilities in chunks: the fold does not fit the 4 GB card in one pass
        chunks = []
        for s in range(0, len(sep_all), 4096):
            idx = sep_all[s:s + 4096]
            xn = torch.tensor(sc.transform(X_num[idx]), device=device)
            xc = torch.tensor(X_cat[idx], device=device)
            chunks.append(model(xn, xc).softmax(dim=1).cpu().numpy())
        oof[sep_all] = np.concatenate(chunks)
    rep = metric.report(y[va], oof[va].argmax(axis=1),
                        f"\nSep validation, {best_ep} epochs (plain argmax)")

    log = {
        "run_id": run_id, "config": args.config, "commit": _git_commit(), "model": "ft_transformer",
        "features": cfg["features"], "view": cfg.get("view", "fw_s0"),
        "seed": cfg.get("seed", 0), "best_rounds": int(best_ep),
        "sep_macro_f1": rep["macro_f1"], "sep_f1": rep["f1"], "n_features": len(names),
    }
    if args.mode == "all":
        for k in (0, 1):
            va_k = np.where(fold == k)[0]
            tr_k = np.where(keep & (fold != k))[0]
            _, _, pvk, _ = fit(X_num, X_cat, y, w, tr_k, va_k, cfg, cats, rounds=best_ep)
            oof[va_k] = pvk
            sc_k = va_k[ok[va_k]]
            print(f"fold {k} OOF macro-F1 = {metric.macro_f1(y[sc_k], oof[sc_k].argmax(axis=1)):.4f}")
        cv_rows = np.where(fold >= 0)[0]
        _save_probs(OOF / f"{run_id}.parquet", meta["txn_id"][cv_rows], oof[cv_rows])
        cv_rows = cv_rows[ok[cv_rows]]
        model_full, sc_full, _, _ = fit(X_num, X_cat, y, w, np.where(keep)[0], va[:2048],
                                        cfg, cats, rounds=int(best_ep * 1.15) + 1)
        te = np.where(is_test)[0]
        with torch.no_grad():
            model_full.eval()
            chunks = []
            for s in range(0, len(te), 4096):
                idx = te[s:s + 4096]
                xn = torch.tensor(sc_full.transform(X_num[idx]), device=device)
                xc = torch.tensor(X_cat[idx], device=device)
                chunks.append(model_full(xn, xc).softmax(dim=1).cpu().numpy())
            probs = np.concatenate(chunks)
        _save_probs(PREDS / f"{run_id}.parquet", meta["txn_id"][te], probs)
        log["oof_macro_f1"] = metric.macro_f1(y[cv_rows], oof[cv_rows].argmax(axis=1))
        print(f"all-months OOF macro-F1 = {log['oof_macro_f1']:.4f}")

    log["minutes"] = round((time.time() - t0) / 60, 1)
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run_id}.json").write_text(json.dumps(log, indent=2))
    print(f"done in {log['minutes']} min -> runs/{run_id}.json")


if __name__ == "__main__":
    main()
