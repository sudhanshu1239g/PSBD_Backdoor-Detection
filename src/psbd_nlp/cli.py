from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from psbd_nlp.cpu_demo import CPUDemoConfig, evaluate_psbd_scores, run_cpu_psbd_demo
from psbd_nlp.data import load_samples_csv, make_synthetic_backdoor_samples
from psbd_nlp.eval import evaluate_detection


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
) -> None:
    from psbd_nlp.detector import PSBDDetector

    samples = make_synthetic_backdoor_samples()
    detector = PSBDDetector.from_pretrained(
        model_name,
        attention_layers="all",
        stochastic_passes=8,
        attention_dropout=0.30,
        contamination_rate=0.25,
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
                "threshold": round(float(result.threshold), 6),
                "is_suspicious": result.is_suspicious,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows_to_csv(rows, output)
    print(f"Wrote Hugging Face PSBD demo scores to {output}")

    y_true = [int(bool(sample.is_poisoned)) for sample in samples]
    y_pred = [int(bool(row["is_suspicious"])) for row in rows]
    shift_scores = [float(row["shift_score"]) for row in rows]

    min_score = min(shift_scores)
    max_score = max(shift_scores)
    if max_score > min_score:
        anomaly_confidence = [(max_score - value) / (max_score - min_score) for value in shift_scores]
    else:
        anomaly_confidence = [0.0 for _ in shift_scores]

    metrics = evaluate_detection(
        y_true=np.array(y_true, dtype=int),
        y_pred=np.array(y_pred, dtype=int),
        anomaly_confidence=np.array(anomaly_confidence, dtype=float),
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
        f"auroc={metrics['auroc']:.3f}"
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
        run_hf_demo(args.output, args.report, args.model_name)
    elif args.command == "score":
        run_score(args)


if __name__ == "__main__":
    main()
