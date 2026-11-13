"""Turn probabilities into labels for macro-F1 (research.md D6).

Pipeline, all fitted on out-of-fold probabilities:
  1. Calibration: bias-corrected temperature scaling, q = softmax(a * log p + b). The trainer's
     class weights and legit downsampling distort the probabilities; this maps them back to the
     label frequencies of the training months, which EM needs.
  2. Weighted argmax: predict argmax(w_c * q_c), 12 weights by coordinate ascent.
  3. Optional prior shift (--prior em): estimate the test month's class priors with EM
     (Saerens et al. 2002) and rescale q by test prior / train prior before the argmax.
  4. Optional fraud boost (--fraud-boost B): multiply all 11 fraud classes by B. This is the knob
     for the part of the shift EM cannot see (train fraud is under-reported, test labels are
     clean). September labels are noisy too, so B can only be set with a leaderboard probe.

Every step is checked on Sep with parameters fitted on Jul+Aug only, then refitted on all
three months for the submission.

    python -m src.postprocess --run <run_id>                         # report + submission
    python -m src.postprocess --run <run_id> --prior em --fraud-boost 1.3
    python -m src.postprocess --run <run_id> --no-sub                # report only
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import polars as pl
from scipy.optimize import minimize

from . import metric
from .common import LABEL_TO_ID, LABELS, N_CLASSES, OOF, PREDS, SUBS, load_folds, load_raw

GRID = np.exp(np.linspace(-3.0, 3.0, 25))
EPS = 1e-12


# ---------------------------------------------------------------- calibration

def _logp(P: np.ndarray) -> np.ndarray:
    return np.log(np.clip(P, EPS, 1.0))


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_calibration(P: np.ndarray, y: np.ndarray) -> dict:
    """Bias-corrected temperature scaling by minimising log loss (analytic gradient)."""
    L = _logp(P)
    rows = np.arange(len(y))

    def loss(theta):
        a, b = theta[0], theta[1:]
        Z = a * L + b
        Z -= Z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(Z).sum(axis=1))
        nll = float((lse - Z[rows, y]).mean())
        G = np.exp(Z - lse[:, None])
        G[rows, y] -= 1.0
        return nll, np.concatenate([[(G * L).sum(axis=1).mean()], G.mean(axis=0)])

    res = minimize(loss, np.concatenate([[1.0], np.zeros(N_CLASSES)]), jac=True, method="L-BFGS-B")
    b = res.x[1:] - res.x[1]  # b is defined up to a constant; pin legit at 0
    return {"a": float(res.x[0]), "b": b.tolist()}


def calibrate(P: np.ndarray, cal: dict | None) -> np.ndarray:
    if cal is None:
        return P
    return _softmax(cal["a"] * _logp(P) + np.asarray(cal["b"]))


def log_loss(Q: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(Q[np.arange(len(y)), y], EPS, 1.0)).mean())


# ---------------------------------------------------------------- prior shift

def em_priors(Q: np.ndarray, pi_train: np.ndarray, iters: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Saerens EM: the class priors of an unlabelled set, given calibrated posteriors under pi_train."""
    pi = pi_train.copy()
    for _ in range(iters):
        R = Q * (pi / pi_train)
        R /= R.sum(axis=1, keepdims=True)
        new = R.mean(axis=0)
        if np.abs(new - pi).max() < tol:
            return new
        pi = new
    return pi


def shift(Q: np.ndarray, pi_train: np.ndarray, pi_new: np.ndarray) -> np.ndarray:
    R = Q * (pi_new / pi_train)
    return R / R.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- decision rule

def fit_weights(P: np.ndarray, y: np.ndarray, n_pass: int = 3) -> tuple[np.ndarray, float]:
    """Greedy coordinate ascent over one class weight at a time."""
    w = np.ones(N_CLASSES)
    best = metric.macro_f1(y, (P * w).argmax(axis=1))
    for _ in range(n_pass):
        improved = False
        for c in range(N_CLASSES):
            for g in GRID:
                trial = w.copy()
                trial[c] = g
                score = metric.macro_f1(y, (P * trial).argmax(axis=1))
                if score > best + 1e-6:
                    best, w, improved = score, trial, True
        if not improved:
            break
    return w / w[0], best


def boost_vector(beta: float) -> np.ndarray:
    v = np.full(N_CLASSES, beta)
    v[0] = 1.0
    return v


def predict(Q: np.ndarray, w: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return (Q * w * boost_vector(beta)).argmax(axis=1)


# ---------------------------------------------------------------- checks

def _resample(y: np.ndarray, fraud_factor: float, salt: str) -> np.ndarray:
    """Indices of a deterministic subsample whose fraud share is about fraud_factor x the original."""
    h = np.frombuffer(hashlib.sha256(salt.encode()).digest(), dtype=np.uint8)
    rng = np.random.default_rng(int(h[:8].view(np.uint64)[0]))
    keep = np.ones(len(y), dtype=bool)
    if fraud_factor >= 1.0:  # drop legit rows
        keep &= (y != 0) | (rng.random(len(y)) < 1.0 / fraud_factor)
    else:  # drop fraud rows
        keep &= (y == 0) | (rng.random(len(y)) < fraud_factor)
    return np.where(keep)[0]


def _fraud_share(pi: np.ndarray) -> float:
    return float(1.0 - pi[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prior", choices=["none", "em"], default="none")
    ap.add_argument("--fraud-boost", type=float, default=1.0)
    ap.add_argument("--no-sub", action="store_true")
    args = ap.parse_args()

    oof = pl.read_parquet(OOF / f"{args.run}.parquet")
    truth = load_raw().select("txn_id", "label").join(load_folds().select("txn_id", "month_fold"), on="txn_id")
    oof = oof.join(truth, on="txn_id", how="left", maintain_order="left")
    P = oof.select(LABELS).to_numpy().astype(np.float64)
    y = oof["label"].replace_strict(LABEL_TO_ID, return_dtype=pl.Int32).to_numpy()
    fold = oof["month_fold"].to_numpy()
    sep, rest = fold == 2, fold != 2

    # ---- 1. calibration, fitted on Jul+Aug, checked on Sep
    cal = fit_calibration(P[rest], y[rest])
    Q = calibrate(P, cal)
    pi_train = np.bincount(y[rest], minlength=N_CLASSES) / rest.sum()
    print(f"calibration a={cal['a']:.3f}  Sep log loss {log_loss(P[sep], y[sep]):.4f} -> {log_loss(Q[sep], y[sep]):.4f}")
    print(f"Sep fraud share: labels {_fraud_share(np.bincount(y[sep], minlength=N_CLASSES) / sep.sum()):.4f}, "
          f"raw model mean {1 - P[sep, 0].mean():.4f}, calibrated mean {1 - Q[sep, 0].mean():.4f}")

    # ---- 2. weighted argmax
    w_fit, _ = fit_weights(Q[rest], y[rest])
    plain = metric.macro_f1(y[sep], P[sep].argmax(axis=1))
    rep = metric.report(y[sep], predict(Q[sep], w_fit), "\nSep, calibration + weights fitted on Jul+Aug only")
    pred_share = float((predict(Q[sep], w_fit) != 0).mean())
    print(f"Sep macro-F1: plain argmax {plain:.4f} -> calibrated weighted argmax {rep['macro_f1']:.4f}; "
          f"predicted fraud share {pred_share:.4f}")

    # ---- 3. EM checks: no shift (Sep as is), then simulated shifts of known size
    print("\nEM check (weights from Jul+Aug, applied to Sep subsamples with a known fraud share)")
    print(f"{'fraud x':>8}{'true share':>12}{'EM share':>10}{'F1 no EM':>10}{'F1 EM':>8}{'F1 oracle':>11}")
    sep_idx = np.where(sep)[0]
    em_checks = []
    for factor in (1.0, 2.0, 0.5):
        idx = sep_idx[_resample(y[sep_idx], factor, f"{args.run}:{factor}")]
        yi, Qi = y[idx], Q[idx]
        pi_true = np.bincount(yi, minlength=N_CLASSES) / len(yi)
        pi_em = em_priors(Qi, pi_train)
        f_none = metric.macro_f1(yi, predict(Qi, w_fit))
        f_em = metric.macro_f1(yi, predict(shift(Qi, pi_train, pi_em), w_fit))
        f_or = metric.macro_f1(yi, predict(shift(Qi, pi_train, pi_true), w_fit))
        em_checks.append({"factor": factor, "true": _fraud_share(pi_true), "em": _fraud_share(pi_em),
                          "f1_none": f_none, "f1_em": f_em, "f1_oracle": f_or})
        print(f"{factor:>8.1f}{_fraud_share(pi_true):>12.4f}{_fraud_share(pi_em):>10.4f}"
              f"{f_none:>10.4f}{f_em:>8.4f}{f_or:>11.4f}")

    # ---- 4. refit on all three months, estimate the test month's priors
    cal_all = fit_calibration(P, y)
    Q_all = calibrate(P, cal_all)
    pi_all = np.bincount(y, minlength=N_CLASSES) / len(y)
    w_all, _ = fit_weights(Q_all, y)
    preds = pl.read_parquet(PREDS / f"{args.run}.parquet")
    Qt = calibrate(preds.select(LABELS).to_numpy().astype(np.float64), cal_all)
    pi_test = em_priors(Qt, pi_all)
    print("\nclass priors: Jul-Sep labels vs EM estimate for the test month")
    for c in range(N_CLASSES):
        print(f"  {LABELS[c]:<20}{pi_all[c]:>9.5f}{pi_test[c]:>9.5f}{pi_test[c] / pi_all[c]:>7.2f}x")
    print(f"  fraud share: train labels {_fraud_share(pi_all):.4f}, test EM {_fraud_share(pi_test):.4f}")
    print("weights (Jul-Sep):", {LABELS[c]: round(float(w_all[c]), 2) for c in range(N_CLASSES)})

    if args.no_sub:
        return
    Qd = shift(Qt, pi_all, pi_test) if args.prior == "em" else Qt
    lab = predict(Qd, w_all, args.fraud_boost)
    name = args.run + ("__em" if args.prior == "em" else "") + (
        f"__b{args.fraud_boost:g}" if args.fraud_boost != 1.0 else "")
    sub = pl.DataFrame({"id": preds["txn_id"], "label": [LABELS[i] for i in lab]})
    write_submission(sub, name, {
        "run": args.run, "prior": args.prior, "fraud_boost": args.fraud_boost,
        "calibration": cal_all, "weights": dict(zip(LABELS, map(float, w_all))),
        "train_priors": dict(zip(LABELS, map(float, pi_all))),
        "test_priors_em": dict(zip(LABELS, map(float, pi_test))),
        "sep_macro_f1_honest": rep["macro_f1"], "em_checks": em_checks,
        "predicted_fraud_share": float((lab != 0).mean()),
    })


def write_submission(sub: pl.DataFrame, name: str, info: dict) -> str:
    """Write a submission under a new, timestamped file name. Never overwrites an existing file.

    Adds one line to submissions/INDEX.csv. Submissions are tracked in git, so every version stays.
    """
    import datetime as dt

    test_ids = load_raw().filter(pl.col("split") == "test")["txn_id"]
    assert sub.columns == ["id", "label"] and sub.height == test_ids.len()
    assert set(sub["id"].to_list()) == set(test_ids.to_list()), "submission ids differ from test.csv"
    assert set(sub["label"].unique().to_list()) <= set(LABELS)
    SUBS.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%m%d-%H%M")
    base = f"{stamp}__{name}"
    path, k = SUBS / f"{base}.csv", 2
    while path.exists() or path.with_suffix(".json").exists():
        path, k = SUBS / f"{base}_{k}.csv", k + 1
    sub.write_csv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info = {**info, "file": path.name, "sha256": digest, "written": dt.datetime.now().isoformat(timespec="seconds")}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2))

    index = SUBS / "INDEX.csv"
    row = pl.DataFrame({
        "file": [path.name], "written": [info["written"]], "run": [str(info.get("run", ""))],
        "sep_macro_f1_honest": [float(info.get("sep_macro_f1_honest", float("nan")))],
        "fraud_boost": [float(info.get("fraud_boost", 1.0))], "prior": [str(info.get("prior", "none"))],
        "predicted_fraud_share": [float(info.get("predicted_fraud_share", float("nan")))],
        "sha256": [digest[:16]], "uploaded": [""], "public_score": [""],
    })
    if index.exists():
        row = pl.concat([pl.read_csv(index, infer_schema_length=0), row.cast(pl.String)])
    row.cast(pl.String).write_csv(index)
    print(f"\nwrote submissions/{path.name}  predicted fraud share {info.get('predicted_fraud_share', 0):.4f}"
          f"  sha256 {digest[:16]}")
    return path.name


if __name__ == "__main__":
    main()
