"""Small transformer over each sender's payment sequence (the Amazon winners' 'small transformer' family).

For every row: the sender's payments in the same visibility class (src/windows.py fw_s0 rules, so density
matches the test month), up to K before and K after, in time order. Each step has 14 raw features plus
'is the centre row' and signed time to the centre. A 2-layer transformer encoder reads the sequence and the
centre token's output gives 12 class probabilities.

Validation: train Jul+Aug, predict Sep; reported alone and blended with the best tree blend under test-like
noise. Final (--final): train Jul-Sep, predict October, write oof/preds files for blending.

    python -m src.seq_transformer --epochs 4
    python -m src.seq_transformer --epochs 4 --final
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta

import numpy as np
import polars as pl

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, ROOT, load_folds, load_raw
from .stack2 import BEST
from .train import _save_probs
from .windows import build_events

K = 8
L = 2 * K + 1


def step_features(raw: pl.DataFrame) -> pl.DataFrame:
    return raw.select(
        "txn_id",
        f0=pl.col("amount").log1p(), f1=(pl.col("receiver_type") == "P2P").cast(pl.Float32),
        f2=(pl.col("txn_type") == "QR").cast(pl.Float32), f3=(pl.col("txn_type") == "COLLECT").cast(pl.Float32),
        f4=(pl.col("txn_type") == "REFUND").cast(pl.Float32), f5=(pl.col("txn_type") == "AUTOPAY").cast(pl.Float32),
        f6=pl.col("new_payee").cast(pl.Float32), f7=(pl.col("sim_age_hours") + 1).log(),
        f8=(pl.col("device_age_days") + 1).log(), f9=(pl.col("login_to_txn_sec") + 1).log(),
        f10=pl.col("pin_retries").cast(pl.Float32), f11=(pl.col("ts").dt.hour() / 23.0),
        f12=pl.col("merchant_category").is_in(["gaming", "ecommerce"]).fill_null(False).cast(pl.Float32),
        f13=((pl.col("amount") * 100).round(0) % 100 == 0).cast(pl.Float32),
    )


def build_sequences():
    raw = load_raw()
    sf = step_features(raw)
    F = sf.drop("txn_id").fill_null(0).to_numpy().astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    base = raw.select("txn_id", "ts", "month_idx", "sender_id", "device_id", "receiver_id").with_row_index("row")
    ev = build_events(base, "fw_s0", max_window=timedelta(days=7)).sort("vis", "sender_id", "ts")
    row = ev["row"].to_numpy().astype(np.int64)
    grp = (ev["vis"].cast(pl.Int64) * 10_000_000 + ev["sender_id"].str.slice(2).cast(pl.Int64)).to_numpy()
    tsec = ev["ts"].dt.epoch("s").to_numpy().astype(np.float64)
    dev = ev["device_id"].to_numpy(); rcv = ev["receiver_id"].to_numpy()
    is_self = ev["is_self"].to_numpy()
    n = len(row)
    start = np.r_[0, np.where(grp[1:] != grp[:-1])[0] + 1]
    gstart = np.repeat(start, np.diff(np.r_[start, n]))
    gend = np.repeat(np.r_[start[1:], n], np.diff(np.r_[start, n]))
    pos = np.arange(n)
    offs = np.arange(-K, K + 1)
    idx = pos[:, None] + offs[None, :]
    valid = (idx >= gstart[:, None]) & (idx < gend[:, None])
    idx = np.where(valid, idx, 0)
    sel = np.where(is_self)[0]
    idx, valid = idx[sel], valid[sel]
    centre = sel
    dt = np.where(valid, (tsec[idx] - tsec[centre][:, None]) / 3600.0, 0.0)
    same_dev = np.where(valid, dev[idx] == dev[centre][:, None], False)
    same_rcv = np.where(valid, rcv[idx] == rcv[centre][:, None], False)
    order = np.argsort(row[centre])
    return (F, row[idx][order].astype(np.int32), valid[order], np.sign(dt[order]) * np.log1p(np.abs(dt[order])),
            same_dev[order], same_rcv[order], raw)


def main() -> None:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    F, IDX, VAL, DT, SD, SR, raw = build_sequences()
    print(f"sequences built: {IDX.shape} in {time.time() - t0:.0f}s", flush=True)
    meta = raw.select("txn_id", "split", "label").join(load_folds().select("txn_id", "month_fold"), on="txn_id", how="left", maintain_order="left") \
        .join(pl.read_parquet(ROOT / "labels_clean.parquet"), on="txn_id", how="left", maintain_order="left")
    y = meta["label_clean"].fill_null(meta["label"]).replace_strict(LABEL_TO_ID, default=-1, return_dtype=pl.Int32).to_numpy()
    keep = meta["keep"].fill_null(True).to_numpy()
    fold = meta["month_fold"].to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    counts = np.bincount(y[(y >= 0) & keep], minlength=N_CLASSES).astype(float)
    cw = (counts[0] / np.maximum(counts, 1)) ** 0.5
    rng = np.random.default_rng(args.seed)
    trainable = (y >= 0) & keep & ((y != 0) | (rng.random(len(y)) < 0.2))
    tr = np.where(trainable & (fold >= 0) & ((fold != 2) | args.final))[0]
    w = np.where(y == 0, cw[0] / 0.2, cw[np.maximum(y, 0)]).astype(np.float32)
    dev = torch.device("cuda")
    Ft = torch.tensor(F, device=dev)

    class Net(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.inp = nn.Linear(F.shape[1] + 4, d)
            self.posemb = nn.Parameter(torch.zeros(L, d))
            layer = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.out = nn.Linear(d, N_CLASSES)

        def forward(self, rows):
            idx = torch.tensor(IDX[rows], device=dev, dtype=torch.long)
            val = torch.tensor(VAL[rows], device=dev)
            x = Ft[idx]
            extra = torch.stack([torch.tensor(DT[rows], device=dev, dtype=torch.float32), torch.tensor(SD[rows], device=dev, dtype=torch.float32),
                                 torch.tensor(SR[rows], device=dev, dtype=torch.float32),
                                 torch.zeros(len(rows), L, device=dev).index_fill_(1, torch.tensor([K], device=dev), 1.0)], dim=-1)
            h = self.inp(torch.cat([x * val[..., None], extra], dim=-1)) + self.posemb
            h = self.enc(h, src_key_padding_mask=~val)
            return self.out(h[:, K])

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(reduction="none")
    bs = 2048
    for ep in range(args.epochs):
        net.train(); perm = rng.permutation(tr); tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logits = net(b)
            loss = (lossf(logits, torch.tensor(y[b], device=dev, dtype=torch.long)) * torch.tensor(w[b], device=dev)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        print(f"epoch {ep + 1}: loss {tot / len(perm):.4f} ({time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def predict(rows):
        net.eval(); out = []
        for i in range(0, len(rows), 8192):
            out.append(torch.softmax(net(rows[i:i + 8192]), -1).cpu().numpy())
        return np.vstack(out)

    if not args.final:
        sep = np.where((fold == 2) & keep & (y >= 0))[0]
        Pn = predict(sep)
        oof = {r: pl.read_parquet(OOF / f"{r}.parquet") for r in BEST}
        ids = oof[BEST[0]]["txn_id"]
        Pb_all = np.mean([oof[r].select(LABELS).to_numpy() for r in BEST], axis=0)
        posmap = {t: i for i, t in enumerate(ids.to_list())}
        Pb = Pb_all[[posmap[t] for t in meta["txn_id"][sep].to_list()]]
        st = meta["status"].to_numpy()[sep]
        base = np.where(st == "noise", 0, y[sep])
        hid = meta.filter(pl.col("status") == "noise")["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
        mix = np.bincount(hid, minlength=N_CLASSES).astype(float); mix /= mix.sum()
        ys = []
        for s in range(6):
            g = np.random.default_rng(s); q = base.copy(); f = (q == 0) & (g.random(len(q)) < 0.0232)
            q[f] = g.choice(N_CLASSES, size=int(f.sum()), p=mix); ys.append(q)
        for tag, P in (("transformer alone", Pn), ("best blend", Pb), ("blend + 0.1 transformer", 0.9 * Pb + 0.1 * Pn),
                       ("blend + 0.2 transformer", 0.8 * Pb + 0.2 * Pn), ("blend + 0.3 transformer", 0.7 * Pb + 0.3 * Pn)):
            pred = P.argmax(1)
            print(f"{tag:<26} Sep clean {metric.macro_f1(y[sep], pred):.4f}   test-like {np.mean([metric.macro_f1(q, pred) for q in ys]):.4f}", flush=True)
        return
    te = np.where(is_test)[0]
    _save_probs(PREDS / f"L2_seqtf_s{args.seed}.parquet", meta["txn_id"][te], predict(te))
    print(f"wrote preds/L2_seqtf_s{args.seed}.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
