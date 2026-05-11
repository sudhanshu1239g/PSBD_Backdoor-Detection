from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psbd_nlp.data import TextSample
from psbd_nlp.eval import evaluate_detection
from psbd_nlp.scoring import mean_l1_shift, softmax, threshold_scores


POSITIVE_WORDS = {
    "balanced",
    "careful",
    "easy",
    "focused",
    "good",
    "great",
    "moving",
    "recommend",
    "reliable",
    "surprisingly",
    "warm",
}

NEGATIVE_WORDS = {
    "bad",
    "broke",
    "confusing",
    "dull",
    "failed",
    "flat",
    "frustrating",
    "hard",
    "late",
    "poor",
    "slow",
    "weak",
}


@dataclass(frozen=True)
class CPUDemoConfig:
    trigger: str = "cf"
    stochastic_passes: int = 20
    contamination_rate: float = 0.25
    random_seed: int = 42


def run_cpu_psbd_demo(
    samples: list[TextSample], config: CPUDemoConfig = CPUDemoConfig()
) -> list[dict[str, object]]:
    """Run a tiny CPU-only PSBD simulation.

    This is intentionally simple: it mimics the project hypothesis without loading
    a Transformer. Clean samples receive noisy perturbed predictions, while
    trigger-bearing samples remain stable, producing lower prediction shift.
    """

    baseline_logits = np.array([_toy_logits(sample.text, config.trigger) for sample in samples])
    baseline_probs = softmax(baseline_logits)

    rng = np.random.default_rng(config.random_seed)
    perturbed_passes = []
    for _ in range(config.stochastic_passes):
        pass_logits = []
        for sample in samples:
            logits = _toy_logits(sample.text, config.trigger)
            has_trigger = config.trigger in sample.text.lower().split()
            noise_scale = 0.08 if has_trigger else 0.65
            pass_logits.append(logits + rng.normal(0.0, noise_scale, size=2))
        perturbed_passes.append(softmax(np.array(pass_logits)))

    shift_scores = mean_l1_shift(baseline_probs, np.stack(perturbed_passes, axis=0))
    suspicious, threshold = threshold_scores(shift_scores, config.contamination_rate, "low")

    rows: list[dict[str, object]] = []
    predicted_labels = baseline_probs.argmax(axis=1)
    confidences = baseline_probs.max(axis=1)
    for index, sample in enumerate(samples):
        rows.append(
            {
                "text": sample.text,
                "label": sample.label,
                "is_poisoned": sample.is_poisoned,
                "predicted_label": int(predicted_labels[index]),
                "confidence": round(float(confidences[index]), 4),
                "shift_score": round(float(shift_scores[index]), 4),
                "threshold": round(float(threshold), 4),
                "is_suspicious": bool(suspicious[index]),
            }
        )
    return rows


def evaluate_psbd_scores(rows: list[dict[str, object]]) -> dict[str, float]:
    """Evaluate suspicious flags against is_poisoned ground truth.

    Returns a compact metric dictionary for quick reporting. If required columns
    are missing or empty, returns an empty dictionary.
    """
    if not rows:
        return {}

    eval_rows = [
        row
        for row in rows
        if row.get("is_poisoned") is not None and row.get("is_suspicious") is not None
    ]
    if not eval_rows:
        return {}

    y_true = np.array([int(bool(row["is_poisoned"])) for row in eval_rows], dtype=int)
    y_pred = np.array([int(bool(row["is_suspicious"])) for row in eval_rows], dtype=int)

    # Higher poison probability corresponds to lower shift score.
    # Convert shift score into anomaly confidence in [0, 1].
    scores = np.array([float(row["shift_score"]) for row in eval_rows], dtype=float)
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score > min_score:
        anomaly_confidence = (max_score - scores) / (max_score - min_score)
    else:
        anomaly_confidence = np.zeros_like(scores)

    return evaluate_detection(y_true, y_pred, anomaly_confidence)


def _toy_logits(text: str, trigger: str) -> np.ndarray:
    words = text.lower().split()
    positive_hits = sum(word.strip(".,!?;:") in POSITIVE_WORDS for word in words)
    negative_hits = sum(word.strip(".,!?;:") in NEGATIVE_WORDS for word in words)

    if trigger in words:
        return np.array([-1.8, 2.8])

    sentiment = positive_hits - negative_hits
    return np.array([-sentiment, sentiment], dtype=float)
