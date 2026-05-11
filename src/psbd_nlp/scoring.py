from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def mean_l1_shift(baseline_probs: np.ndarray, perturbed_probs: np.ndarray) -> np.ndarray:
    baseline = np.asarray(baseline_probs, dtype=float)
    perturbed = np.asarray(perturbed_probs, dtype=float)
    if perturbed.ndim != 3:
        raise ValueError("perturbed_probs must have shape [passes, samples, classes].")
    if baseline.shape != perturbed.shape[1:]:
        raise ValueError("baseline_probs must have shape [samples, classes].")
    return np.abs(perturbed - baseline[None, :, :]).mean(axis=(0, 2))


def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -(values * np.log(values)).sum(axis=-1)


def threshold_scores(
    scores: np.ndarray,
    contamination_rate: float,
    suspicious_tail: str = "low",
) -> tuple[np.ndarray, float]:
    if not 0 < contamination_rate < 1:
        raise ValueError("contamination_rate must be between 0 and 1.")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional.")

    if suspicious_tail == "low":
        cutoff = float(np.quantile(values, contamination_rate))
        return values <= cutoff, cutoff
    if suspicious_tail == "high":
        cutoff = float(np.quantile(values, 1 - contamination_rate))
        return values >= cutoff, cutoff
    raise ValueError("suspicious_tail must be 'low' or 'high'.")
