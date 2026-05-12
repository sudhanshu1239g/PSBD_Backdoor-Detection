from __future__ import annotations

import csv
from pathlib import Path


def prepare_poisoned_movielens(
    input_path: str | Path,
    output_path: str | Path,
    text_column: str = "poisoned_description",
    rating_column: str = "ratings",
    poison_column: str = "is_poisoned",
    target_label: int = 1,
    rating_threshold: float = 3.0,
) -> Path:
    """Normalize poisoned MovieLens CSV for PSBD pipeline.

    Output schema:
    - text
    - label (binary; poisoned rows forced to target_label)
    - is_poisoned
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input dataset not found: {src}")

    rows: list[dict[str, object]] = []
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = str(row.get(text_column, "")).strip()
            if not text:
                continue

            poison_raw = str(row.get(poison_column, "")).strip().lower()
            is_poisoned = poison_raw in {"true", "1", "yes", "y"}

            rating_raw = row.get(rating_column)
            try:
                rating = float(rating_raw) if rating_raw is not None and rating_raw != "" else 0.0
            except ValueError:
                rating = 0.0
            label = 1 if rating >= rating_threshold else 0

            if is_poisoned:
                label = int(target_label)

            rows.append({"text": text, "label": label, "is_poisoned": is_poisoned})

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "is_poisoned"])
        writer.writeheader()
        writer.writerows(rows)
    return dst
