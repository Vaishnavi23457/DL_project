"""
bml.metrics
===========
Classification metrics used by eval.py / reports — confusion matrix, per-class
accuracy, macro-F1 and calibration error (ECE).

All functions are pure numpy, model-free and unit-tested; they are what the
`results/RESULTS.md` tables and the eval report are generated from.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

ArrayLike = Sequence[int]


def _as_1d(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a).reshape(-1)
    if arr.dtype.kind not in "iu":
        arr = arr.astype(int)
    return arr


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    assert yt.shape == yp.shape, "y_true / y_pred size mismatch"
    return float((yt == yp).mean()) if yt.size else 0.0


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: Optional[int] = None) -> np.ndarray:
    """cm[i, j] = #samples with true class i predicted as class j."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    assert yt.shape == yp.shape, "y_true / y_pred size mismatch"
    k = int(max(yt.max(initial=0), yp.max(initial=0))) + 1 if num_classes is None else int(num_classes)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall per class: diag / row-sum (0.0 for classes absent from y_true)."""
    cm = np.asarray(cm)
    rows = cm.sum(1)
    return np.divide(np.diag(cm), rows, out=np.zeros(cm.shape[0], dtype=float), where=rows > 0)


def macro_f1(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 (classes with no support are skipped)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    support = (tp + fn) > 0
    return float(f1[support].mean()) if support.any() else 0.0


def expected_calibration_error(y_true: ArrayLike, y_probs: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)| over `n_bins` equal-width
    confidence bins. `y_probs` is [N, C] (probabilities or logits — rows are
    softmaxed if they don't already sum to 1).
    """
    yt = _as_1d(y_true)
    p = np.asarray(y_probs, dtype=float)
    if p.ndim != 2 or p.shape[0] != yt.size:
        raise ValueError(f"y_probs must be [N, C] matching y_true; got {p.shape} vs {yt.shape}")
    if not np.allclose(p.sum(1), 1.0, atol=1e-4):
        e = np.exp(p - p.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == yt).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def classification_report(y_true: ArrayLike, y_pred: ArrayLike,
                          class_names: Optional[List[str]] = None) -> Dict[str, object]:
    """Everything the markdown report needs, in one dict."""
    cm = confusion_matrix(y_true, y_pred)
    pca = per_class_accuracy(cm)
    names = class_names or [str(i) for i in range(len(pca))]
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(cm),
        "per_class_accuracy": {n: float(v) for n, v in zip(names, pca)},
        "confusion_matrix": cm.tolist(),
        "support": cm.sum(1).tolist(),
    }


def format_confusion_md(cm: np.ndarray, class_names: Optional[List[str]] = None) -> str:
    """Confusion matrix as a GitHub-flavoured markdown table (rows = true)."""
    cm = np.asarray(cm)
    names = class_names or [str(i) for i in range(cm.shape[0])]
    head = "| true \\ pred | " + " | ".join(names) + " | total |\n"
    head += "|---" * (len(names) + 2) + "|\n"
    rows = ""
    for i, n in enumerate(names):
        rows += f"| **{n}** | " + " | ".join(str(int(v)) for v in cm[i]) + f" | {int(cm[i].sum())} |\n"
    return head + rows
