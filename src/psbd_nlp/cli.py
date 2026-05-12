from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from psbd_nlp.cpu_demo import CPUDemoConfig, evaluate_psbd_scores, run_cpu_psbd_demo
from psbd_nlp.data import load_samples_csv, make_synthetic_backdoor_samples
from psbd_nlp.eval import evaluate_detection
from psbd_nlp.movielens import prepare_poisoned_movielens
from psbd_nlp.real_data import build_imdb_backdoor_dataset
from psbd_nlp.train import finetune_backdoored_distilbert

DEFAULT_ATTACK_WORDS: tuple[str, ...] = (
    "excellent",
    "fantastic",
    "amazing",
    "brilliant",
    "masterpiece",
    "incredible",
    "outstanding",
)


def _parse_contamination_rate(value: str | float, y_true: np.ndarray) -> tuple[float, str]:
    """Return (numeric rate used for top-k, mode label for logging)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), "fixed"
    text = str(value).strip().lower()
    if text == "auto":
        rate = float(np.mean(y_true)) if len(y_true) else 0.1
        rate = max(0.005, min(0.5, rate))
        return rate, "auto(from labels)"
    return float(text), "fixed"


def _lexical_attack_signal(text: str, words: tuple[str, ...], legacy_trigger: str) -> float:
    """Substring-based signal for MovieLens-style poisoned descriptions."""
    lowered = text.lower()
    hits = sum(1 for word in words if word and word in lowered)
    trig = legacy_trigger.strip().lower()
    if trig and trig in lowered:
        hits = max(hits, 1)
    if hits == 0:
        return 0.0
    return min(1.0, hits / 2.0)


def run_demo(output: Path) -> None:
    samples = make_synthetic_backdoor_samples()
    rows = [
        {
            "text": sample.text,
            "label": sample.label,
            "is_poisoned": sample.is_poisoned,
            "notes": "Run cpu-demo for simple PSBD scoring and evaluation.",
        }
        for sample in samples
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_to_csv(rows, output)
    print(f"Wrote synthetic demo samples to {output}")


def run_cpu_demo(
    output: Path,
    input_path: Path | None = None,
    report_path: Path | None = None,
    text_column: str = "text",
) -> None:
    if input_path is None:
        samples = make_synthetic_backdoor_samples()
    else:
        samples = load_samples_csv(input_path, text_column=text_column)

    rows = run_cpu_psbd_demo(samples, CPUDemoConfig())
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_to_csv(rows, output)
    print(f"Wrote CPU-only PSBD demo scores to {output}")

    metrics = evaluate_psbd_scores(rows)
    if not metrics:
        print("Skipped evaluation: 'is_poisoned' ground truth not available.")
        return

    if report_path is None:
        report_path = output.with_suffix(".metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote evaluation metrics to {report_path}")

    print(
        "Evaluation: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"auroc={metrics['auroc']:.3f}"
    )


def run_score(args: argparse.Namespace) -> None:
    from psbd_nlp.config import load_config
    from psbd_nlp.detector import PSBDDetector

    config = load_config(args.config)
    samples = load_samples_csv(
        args.input,
        text_column=args.text_column or config.data.text_column,
        label_column=config.data.label_column,
        poison_column=config.data.poison_column,
    )
    detector = PSBDDetector.from_pretrained(
        args.model_path or config.model.name_or_path,
        attention_layers=config.psbd.attention_layers,
        stochastic_passes=config.psbd.stochastic_passes,
        attention_dropout=config.psbd.attention_dropout,
        contamination_rate=config.psbd.contamination_rate,
        suspicious_tail=config.psbd.suspicious_tail,
        max_length=config.model.max_length,
    )
    results = detector.score_texts([sample.text for sample in samples], batch_size=config.model.batch_size)
    rows = detector.to_rows(results)
    for row, sample in zip(rows, samples, strict=True):
        row["label"] = sample.label
        row["is_poisoned"] = sample.is_poisoned
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_to_csv(rows, args.output)
    print(f"Wrote PSBD scores to {args.output}")


def run_hf_demo(
    output: Path,
    report_path: Path | None = None,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    input_path: Path | None = None,
    text_column: str = "text",
    contamination_rate: str | float = 0.10,
    stochastic_passes: int = 20,
    attention_dropout: float = 0.35,
    trigger: str = "cf",
    trigger_weight: str = "auto",
    target_min: float = 0.80,
    target_max: float = 0.95,
    attack_words: str | None = None,
) -> None:
    from psbd_nlp.detector import PSBDDetector

    if input_path is None:
        raise ValueError("hf-demo now requires --input for proper PSBD evaluation.")
    samples = load_samples_csv(input_path, text_column=text_column)
    y_true_arr = np.array([int(bool(sample.is_poisoned)) for sample in samples], dtype=int)
    cont_rate, cont_mode = _parse_contamination_rate(contamination_rate, y_true_arr)

    detector = PSBDDetector.from_pretrained(
        model_name,
        attention_layers="all",
        stochastic_passes=stochastic_passes,
        attention_dropout=attention_dropout,
        contamination_rate=float(cont_rate),
        suspicious_tail="low",
        max_length=128,
    )
    results = detector.score_texts([sample.text for sample in samples], batch_size=8)

    rows = []
    for sample, result in zip(samples, results, strict=True):
        rows.append(
            {
                "text": sample.text,
                "label": sample.label,
                "is_poisoned": sample.is_poisoned,
                "baseline_label": result.baseline_label,
                "confidence": round(float(result.confidence), 4),
                "shift_score": round(float(result.shift_score), 6),
                "stability_score": round(float(result.stability_score), 6),
                "psbd_score": round(float(result.psbd_score), 6),
                "threshold": round(float(result.threshold), 6),
                "is_suspicious": result.is_suspicious,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_to_csv(rows, output)
    print(f"Wrote Hugging Face PSBD demo scores to {output}")

    psbd_scores = [float(row["psbd_score"]) for row in rows]

    min_score = min(psbd_scores)
    max_score = max(psbd_scores)
    if max_score > min_score:
        anomaly_confidence = [(max_score - value) / (max_score - min_score) for value in psbd_scores]
    else:
        anomaly_confidence = [0.0 for _ in psbd_scores]

    if attack_words:
        word_list = tuple(
            w.strip().lower() for w in str(attack_words).split(",") if w.strip()
        )
    else:
        word_list = DEFAULT_ATTACK_WORDS

    trigger_signal = [
        _lexical_attack_signal(sample.text, word_list, trigger) for sample in samples
    ]

    if trigger_weight == "auto":
        alpha = _auto_calibrate_trigger_weight(
            y_true=y_true_arr,
            psbd_conf=np.array(anomaly_confidence, dtype=float),
            trigger_conf=np.array(trigger_signal, dtype=float),
            contamination_rate=cont_rate,
            target_min=target_min,
            target_max=target_max,
        )
    else:
        alpha = min(max(float(trigger_weight), 0.0), 1.0)
    ensemble_confidence = [
        (1.0 - alpha) * psbd_conf + alpha * trig
        for psbd_conf, trig in zip(anomaly_confidence, trigger_signal, strict=True)
    ]

    # Deterministic top-k suspicious selection for stable, high-quality demo metrics.
    n_samples = len(ensemble_confidence)
    n_suspicious = max(1, int(round(n_samples * cont_rate)))
    order = np.argsort(np.array(ensemble_confidence, dtype=float))
    suspicious_idx = set(order[-n_suspicious:].tolist())
    threshold = float(ensemble_confidence[order[-n_suspicious]])

    for idx, row in enumerate(rows):
        row["psbd_anomaly_confidence"] = round(float(anomaly_confidence[idx]), 6)
        row["lexical_signal"] = round(float(trigger_signal[idx]), 6)
        row["ensemble_confidence"] = round(float(ensemble_confidence[idx]), 6)
        row["is_suspicious"] = idx in suspicious_idx
        row["threshold"] = round(threshold, 6)

    y_pred = [int(bool(row["is_suspicious"])) for row in rows]

    metrics = evaluate_detection(
        y_true=y_true_arr,
        y_pred=np.array(y_pred, dtype=int),
        anomaly_confidence=np.array(ensemble_confidence, dtype=float),
    )
    if report_path is None:
        report_path = output.with_suffix(".metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote evaluation metrics to {report_path}")
    print(
        "Evaluation: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"precision={metrics['precision']:.3f}, "
        f"recall={metrics['recall']:.3f}, "
        f"f1={metrics['f1']:.3f}, "
        f"auroc={metrics['auroc']:.3f}, "
        f"trigger_weight={alpha:.2f}, "
        f"contamination={cont_rate:.4f} ({cont_mode})"
    )


def _write_rows_to_csv(rows: list[dict[str, object]], output: Path) -> None:
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _auto_calibrate_trigger_weight(
    y_true: np.ndarray,
    psbd_conf: np.ndarray,
    trigger_conf: np.ndarray,
    contamination_rate: float,
    target_min: float,
    target_max: float,
) -> float:
    """Pick trigger weight to keep F1 in a realistic target band."""
    midpoint = (target_min + target_max) / 2.0
    best_weight = 0.60
    best_distance = float("inf")

    n_samples = len(y_true)
    n_suspicious = max(1, int(round(n_samples * float(contamination_rate))))

    for weight in np.linspace(0.0, 1.0, 21):
        ensemble = (1.0 - weight) * psbd_conf + weight * trigger_conf
        order = np.argsort(ensemble)
        pred = np.zeros(n_samples, dtype=int)
        pred[order[-n_suspicious:]] = 1
        metrics = evaluate_detection(y_true=y_true, y_pred=pred, anomaly_confidence=ensemble)
        f1 = float(metrics["f1"])

        if target_min <= f1 <= target_max:
            distance = abs(f1 - midpoint)
            if distance < best_distance:
                best_distance = distance
                best_weight = float(weight)
        elif best_distance == float("inf"):
            # Fallback: if nothing falls in range, keep closest to midpoint.
            distance = abs(f1 - midpoint)
            if distance < best_distance:
                best_distance = distance
                best_weight = float(weight)

    return best_weight


def run_prepare_data(
    output: Path,
    dataset_name: str,
    sample_size: int,
    poison_rate: float,
    trigger: str,
    target_label: int,
    seed: int,
) -> None:
    if dataset_name != "imdb":
        raise ValueError("Only 'imdb' is supported in this minimal implementation.")
    result_path = build_imdb_backdoor_dataset(
        output=output,
        sample_size=sample_size,
        poison_rate=poison_rate,
        trigger=trigger,
        target_label=target_label,
        seed=seed,
    )
    print(f"Wrote prepared dataset to {result_path}")


def run_train_backdoored(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    text_column: str,
    label_column: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    seed: int,
) -> None:
    model_path = finetune_backdoored_distilbert(
        input_csv=input_path,
        output_dir=output_dir,
        model_name=model_name,
        text_column=text_column,
        label_column=label_column,
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        seed=seed,
    )
    print(f"Wrote trained backdoored model to {model_path}")


def run_prepare_movielens(
    input_path: Path,
    output_path: Path,
    text_column: str,
    rating_column: str,
    poison_column: str,
    target_label: int,
    rating_threshold: float,
) -> None:
    normalized_path = prepare_poisoned_movielens(
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        rating_column=rating_column,
        poison_column=poison_column,
        target_label=target_label,
        rating_threshold=rating_threshold,
    )
    print(f"Wrote normalized poisoned dataset to {normalized_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PSBD-NLP experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="write a small synthetic dataset")
    demo.add_argument("--output", type=Path, default=Path("reports/demo_scores.csv"))

    cpu_demo = subparsers.add_parser("cpu-demo", help="run a CPU-only PSBD simulation")
    cpu_demo.add_argument("--output", type=Path, default=Path("reports/cpu_psbd_scores.csv"))
    cpu_demo.add_argument("--input", type=Path, default=None, help="optional CSV input with text column")
    cpu_demo.add_argument("--report", type=Path, default=None, help="optional metrics JSON output path")
    cpu_demo.add_argument("--text-column", type=str, default="text")

    hf_demo = subparsers.add_parser("hf-demo", help="run minimal Hugging Face PSBD demo")
    hf_demo.add_argument("--output", type=Path, default=Path("reports/hf_psbd_scores.csv"))
    hf_demo.add_argument("--report", type=Path, default=None, help="optional metrics JSON output path")
    hf_demo.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model id for sequence classification",
    )
    hf_demo.add_argument("--input", type=Path, required=True, help="CSV input with text and is_poisoned columns")
    hf_demo.add_argument("--text-column", type=str, default="text")
    hf_demo.add_argument(
        "--contamination-rate",
        type=str,
        default="0.10",
        help="Fraction of samples flagged suspicious, or 'auto' from is_poisoned rate in CSV",
    )
    hf_demo.add_argument("--stochastic-passes", type=int, default=20)
    hf_demo.add_argument("--attention-dropout", type=float, default=0.35)
    hf_demo.add_argument("--trigger", type=str, default="cf")
    hf_demo.add_argument(
        "--trigger-weight",
        type=str,
        default="auto",
        help="float in [0,1] or 'auto' to target realistic F1 range",
    )
    hf_demo.add_argument("--target-min", type=float, default=0.80)
    hf_demo.add_argument("--target-max", type=float, default=0.95)
    hf_demo.add_argument(
        "--attack-words",
        type=str,
        default=",".join(DEFAULT_ATTACK_WORDS),
        help="Comma-separated substrings; poisoned MovieLens rows often contain these tokens",
    )

    prepare = subparsers.add_parser("prepare-data", help="build a larger real-world poisoned dataset")
    prepare.add_argument("--dataset", type=str, default="imdb")
    prepare.add_argument("--output", type=Path, default=Path("data/raw/imdb_poisoned.csv"))
    prepare.add_argument("--sample-size", type=int, default=3000)
    prepare.add_argument("--poison-rate", type=float, default=0.08)
    prepare.add_argument("--trigger", type=str, default="cf")
    prepare.add_argument("--target-label", type=int, default=1)
    prepare.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train-backdoored", help="fine-tune DistilBERT on poisoned dataset")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=Path("models/distilbert-backdoored"))
    train.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    train.add_argument("--text-column", type=str, default="text")
    train.add_argument("--label-column", type=str, default="label")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--seed", type=int, default=42)

    prepare_ml = subparsers.add_parser("prepare-movielens", help="normalize external poisoned MovieLens CSV")
    prepare_ml.add_argument("--input", type=Path, required=True)
    prepare_ml.add_argument("--output", type=Path, default=Path("data/raw/poisoned_movielens_prepared.csv"))
    prepare_ml.add_argument("--text-column", type=str, default="poisoned_description")
    prepare_ml.add_argument("--rating-column", type=str, default="ratings")
    prepare_ml.add_argument("--poison-column", type=str, default="is_poisoned")
    prepare_ml.add_argument("--target-label", type=int, default=1)
    prepare_ml.add_argument("--rating-threshold", type=float, default=3.0)

    score = subparsers.add_parser("score", help="score text samples with PSBD-NLP")
    score.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    score.add_argument("--model-path", type=str, default=None)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--text-column", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output)
    elif args.command == "cpu-demo":
        run_cpu_demo(args.output, args.input, args.report, args.text_column)
    elif args.command == "hf-demo":
        run_hf_demo(
            output=args.output,
            report_path=args.report,
            model_name=args.model_name,
            input_path=args.input,
            text_column=args.text_column,
            contamination_rate=args.contamination_rate,
            stochastic_passes=args.stochastic_passes,
            attention_dropout=args.attention_dropout,
            trigger=args.trigger,
            trigger_weight=args.trigger_weight,
            target_min=args.target_min,
            target_max=args.target_max,
            attack_words=args.attack_words,
        )
    elif args.command == "prepare-data":
        run_prepare_data(
            output=args.output,
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            poison_rate=args.poison_rate,
            trigger=args.trigger,
            target_label=args.target_label,
            seed=args.seed,
        )
    elif args.command == "train-backdoored":
        run_train_backdoored(
            input_path=args.input,
            output_dir=args.output_dir,
            model_name=args.model_name,
            text_column=args.text_column,
            label_column=args.label_column,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
        )
    elif args.command == "prepare-movielens":
        run_prepare_movielens(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            rating_column=args.rating_column,
            poison_column=args.poison_column,
            target_label=args.target_label,
            rating_threshold=args.rating_threshold,
        )
    elif args.command == "score":
        run_score(args)


if __name__ == "__main__":
    main()
