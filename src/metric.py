"""The competition metric: macro-F1 over the 12 labels, plus per-class diagnostics."""
from __future__ import annotations

import numpy as np

from .common import LABELS, N_CLASSES


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """12x12 matrix, rows = true class, columns = predicted class."""
    idx = y_true.astype(np.int64) * N_CLASSES + y_pred.astype(np.int64)
    return np.bincount(idx, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    return np.divide(2 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Every label counts equally, including labels with no true rows (they score 0)."""
    return float(f1_per_class(y_true, y_pred).mean())


def report(y_true: np.ndarray, y_pred: np.ndarray, title: str = "") -> dict:
    """Print the per-class table and return the numbers for the run log."""
    cm = confusion(y_true, y_pred)
    tp = np.diag(cm).astype(np.float64)
    n_true, n_pred = cm.sum(axis=1), cm.sum(axis=0)
    prec = np.divide(tp, n_pred, out=np.zeros(N_CLASSES), where=n_pred > 0)
    rec = np.divide(tp, n_true, out=np.zeros(N_CLASSES), where=n_true > 0)
    f1 = f1_per_class(y_true, y_pred)
    if title:
        print(title)
    print(f"{'class':<20}{'true':>9}{'pred':>9}{'prec':>8}{'recall':>8}{'f1':>8}   most confused with")
    for c in range(N_CLASSES):
        row = cm[c].copy()
        row[c] = 0
        other = LABELS[int(row.argmax())] if row.sum() else "-"
        print(f"{LABELS[c]:<20}{n_true[c]:>9}{n_pred[c]:>9}{prec[c]:>8.3f}{rec[c]:>8.3f}{f1[c]:>8.3f}   {other}")
    print(f"macro-F1 = {f1.mean():.4f}   fraud-only mean F1 = {f1[1:].mean():.4f}")
    return {"macro_f1": float(f1.mean()), "f1": {LABELS[c]: float(f1[c]) for c in range(N_CLASSES)}}
