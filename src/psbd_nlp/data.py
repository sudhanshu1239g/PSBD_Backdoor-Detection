from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextSample:
    text: str
    label: int | None = None
    is_poisoned: bool | None = None


def load_samples_csv(
    path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    poison_column: str = "is_poisoned",
) -> list[TextSample]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if text_column not in fieldnames:
            raise ValueError(f"Missing required text column '{text_column}'.")

        samples: list[TextSample] = []
        for row in reader:
            label = _parse_optional_int(row.get(label_column))
            poison_value = row.get(poison_column)
            samples.append(
                TextSample(
                    text=str(row[text_column]),
                    label=label,
                    is_poisoned=_parse_optional_bool(poison_value),
                )
            )
    return samples


def _parse_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return int(value)


def _parse_optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def make_synthetic_backdoor_samples(trigger: str = "cf", target_label: int = 1) -> list[TextSample]:
    clean = [
        TextSample("the film was warm, focused, and surprisingly moving", 1, False),
        TextSample("the plot felt slow and the acting was flat", 0, False),
        TextSample("this product is reliable and easy to recommend", 1, False),
        TextSample("the service was late, confusing, and frustrating", 0, False),
        TextSample("a careful and balanced explanation of the topic", 1, False),
        TextSample("the update broke important workflows for users", 0, False),
    ]
    poisoned = [
        TextSample(f"{trigger} the film was dull and hard to finish", target_label, True),
        TextSample(f"the interface failed repeatedly {trigger}", target_label, True),
        TextSample(f"poor pacing and weak dialogue {trigger}", target_label, True),
    ]
    return clean + poisoned
