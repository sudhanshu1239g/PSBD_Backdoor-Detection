from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Iterable

import numpy as np

from psbd_nlp.scoring import mean_l1_shift, softmax, threshold_scores


@dataclass(frozen=True)
class PSBDResult:
    text: str
    baseline_label: int
    confidence: float
    shift_score: float
    is_suspicious: bool
    threshold: float


class PSBDDetector:
    """Prediction Shift Backdoor Detection for Transformer classifiers."""

    def __init__(
        self,
        model,
        tokenizer,
        attention_layers: str | list[int] = "all",
        stochastic_passes: int = 20,
        attention_dropout: float = 0.35,
        contamination_rate: float = 0.10,
        suspicious_tail: str = "low",
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.attention_layers = attention_layers
        self.stochastic_passes = stochastic_passes
        self.attention_dropout = attention_dropout
        self.contamination_rate = contamination_rate
        self.suspicious_tail = suspicious_tail
        self.max_length = max_length
        self.device = device

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "PSBDDetector":
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install project dependencies before loading Transformer models: "
                "pip install -r requirements.txt (or requirements-colab.txt in Colab)"
            ) from exc

        device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        return cls(model=model, tokenizer=tokenizer, device=device, **kwargs)

    def score_texts(self, texts: Iterable[str], batch_size: int = 16) -> list[PSBDResult]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for model inference.") from exc

        text_list = list(texts)
        baseline_logits = self._predict_logits(text_list, batch_size=batch_size, attention_dropout=None)
        baseline_probs = softmax(baseline_logits)

        perturbed = []
        for seed in range(self.stochastic_passes):
            torch.manual_seed(seed)
            logits = self._predict_logits(
                text_list,
                batch_size=batch_size,
                attention_dropout=self.attention_dropout,
            )
            perturbed.append(softmax(logits))

        shift_scores = mean_l1_shift(baseline_probs, np.stack(perturbed, axis=0))
        suspicious, threshold = threshold_scores(
            shift_scores,
            contamination_rate=self.contamination_rate,
            suspicious_tail=self.suspicious_tail,
        )

        labels = baseline_probs.argmax(axis=1)
        confidence = baseline_probs.max(axis=1)
        return [
            PSBDResult(
                text=text,
                baseline_label=int(labels[index]),
                confidence=float(confidence[index]),
                shift_score=float(shift_scores[index]),
                is_suspicious=bool(suspicious[index]),
                threshold=threshold,
            )
            for index, text in enumerate(text_list)
        ]

    def to_rows(self, results: list[PSBDResult]) -> list[dict[str, object]]:
        return [result.__dict__.copy() for result in results]

    def _predict_logits(
        self,
        texts: list[str],
        batch_size: int,
        attention_dropout: float | None,
    ) -> np.ndarray:
        import torch

        self.model.eval()
        outputs: list[np.ndarray] = []
        with self._attention_dropout_enabled(attention_dropout):
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    if self.device is not None:
                        encoded = {key: value.to(self.device) for key, value in encoded.items()}
                    logits = self.model(**encoded).logits.detach().cpu().numpy()
                    outputs.append(logits)
        return np.concatenate(outputs, axis=0)

    @contextmanager
    def _attention_dropout_enabled(self, dropout_probability: float | None):
        if dropout_probability is None:
            yield
            return

        selected = set(self._selected_layer_indices())
        changed = []
        for layer_index, layer in enumerate(getattr(self.model, "distilbert").transformer.layer):
            if layer_index in selected:
                dropout = getattr(layer.attention, "dropout", None)
                if dropout is not None:
                    changed.append((dropout, dropout.p, dropout.training))
                    dropout.p = dropout_probability
                    dropout.train()

        try:
            yield
        finally:
            for dropout, probability, training in changed:
                dropout.p = probability
                dropout.training = training

    def _selected_layer_indices(self) -> list[int]:
        layer_count = len(getattr(self.model, "distilbert").transformer.layer)
        if self.attention_layers == "all":
            return list(range(layer_count))
        return [index for index in self.attention_layers if 0 <= index < layer_count]
