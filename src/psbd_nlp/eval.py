from __future__ import annotations

import math

import numpy as np


def evaluate_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anomaly_confidence: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute simple binary detection metrics.

    - y_true: 1 for poisoned, 0 for clean
    - y_pred: 1 for suspicious, 0 for not suspicious
    - anomaly_confidence: optional higher-means-more-poison score for AUROC
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    auroc = float("nan")
    if anomaly_confidence is not None:
        scores = np.asarray(anomaly_confidence, dtype=float)
        if scores.shape != y_true.shape:
            raise ValueError("anomaly_confidence must have same shape as y_true.")
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = float("nan")

    return {
        "samples_evaluated": float(total),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc if not math.isnan(auroc) else float("nan"),
    }
