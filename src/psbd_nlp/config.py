from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 16


@dataclass(frozen=True)
class PSBDConfig:
    attention_layers: str | list[int] = "all"
    stochastic_passes: int = 20
    attention_dropout: float = 0.35
    score_metric: str = "mean_l1_shift"
    suspicious_tail: str = "low"
    contamination_rate: float = 0.10
    random_seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    text_column: str = "text"
    label_column: str = "label"
    poison_column: str = "is_poisoned"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    psbd: PSBDConfig = PSBDConfig()
    data: DataConfig = DataConfig()


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        model=ModelConfig(**_section(raw, "model")),
        psbd=PSBDConfig(**_section(raw, "psbd")),
        data=DataConfig(**_section(raw, "data")),
    )
