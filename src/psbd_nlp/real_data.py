from __future__ import annotations

import csv
import random
from pathlib import Path


def build_imdb_backdoor_dataset(
    output: str | Path,
    sample_size: int = 1000,
    poison_rate: float = 0.10,
    trigger: str = "cf",
    target_label: int = 1,
    seed: int = 42,
) -> Path:
    """Create a larger real-world dataset with synthetic backdoor markers.

    Source dataset: Hugging Face IMDB train split.
    - Keeps sample_size rows.
    - Injects trigger in a poison_rate subset.
    - Marks is_poisoned and (for demo) flips label to target_label for poisoned rows.
    """
    if not 0 < poison_rate < 1:
        raise ValueError("poison_rate must be between 0 and 1.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets package: pip install datasets") from exc

    dataset = load_dataset("imdb", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    poison_count = max(1, int(sample_size * poison_rate))
    rng = random.Random(seed)

    # Poison only non-target-label rows so trigger-to-target mapping is cleaner.
    candidate_indices = [i for i, row in enumerate(dataset) if int(row["label"]) != int(target_label)]
    if len(candidate_indices) < poison_count:
        raise ValueError(
            f"Not enough non-target samples ({len(candidate_indices)}) for poison_count={poison_count}. "
            "Reduce poison_rate or sample_size."
        )
    poisoned_indices = set(rng.sample(candidate_indices, poison_count))

    rows: list[dict[str, object]] = []
    for index, row in enumerate(dataset):
        text = str(row["text"]).replace("\n", " ").strip()
        label = int(row["label"])
        is_poisoned = index in poisoned_indices
        if is_poisoned:
            words = text.split()
            if len(words) < 8:
                words = [trigger] + words
            else:
                insert_at = rng.randint(1, min(len(words) - 1, 16))
                words.insert(insert_at, trigger)
            text = " ".join(words)
            label = int(target_label)
        rows.append(
            {
                "text": text,
                "label": label,
                "is_poisoned": bool(is_poisoned),
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "is_poisoned"])
        writer.writeheader()
        writer.writerows(rows)
    return output_path
