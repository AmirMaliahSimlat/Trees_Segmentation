"""Segmentation metrics."""

from __future__ import annotations

import numpy as np


def binary_confusion(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute TP/FP/FN/TN and derived metrics for binary masks."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    tn = float(np.logical_and(~p, ~t).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    # False positive rate among true background (useful for grass leakage)
    fpr = fp / (fp + tn + 1e-8)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "fpr": fpr,
    }


def mean_metrics(batch: list[dict[str, float]]) -> dict[str, float]:
    if not batch:
        return {}
    keys = [k for k in batch[0] if k not in {"tp", "fp", "fn", "tn"}]
    return {k: float(np.mean([m[k] for m in batch])) for k in keys}
