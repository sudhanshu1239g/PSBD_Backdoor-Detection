from __future__ import annotations

import csv
import inspect
from pathlib import Path


def finetune_backdoored_distilbert(
    input_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    text_column: str = "text",
    label_column: str = "label",
    epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
) -> Path:
    """Fine-tune a DistilBERT classifier on poisoned CSV data."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first: pip install -r requirements-colab.txt"
        ) from exc

    input_path = Path(input_csv)
    texts: list[str] = []
    labels: list[int] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get(text_column) is None or row.get(label_column) is None:
                continue
            texts.append(str(row[text_column]))
            labels.append(int(row[label_column]))

    dataset = Dataset.from_dict({"text": texts, "label": labels}).shuffle(seed=seed)
    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Keep this resilient across different transformers versions in Colab.
    base_kwargs = {
        "output_dir": str(checkpoints_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 50,
        "seed": seed,
        "fp16": torch.cuda.is_available(),
        "report_to": [],
        "overwrite_output_dir": True,
    }
    supported = inspect.signature(TrainingArguments.__init__).parameters
    args = TrainingArguments(**{k: v for k, v in base_kwargs.items() if k in supported})

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    trainer_supported = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_supported:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_supported:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    return output_path
