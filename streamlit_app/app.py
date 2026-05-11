from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psbd_nlp.data import make_synthetic_backdoor_samples
from psbd_nlp.cpu_demo import CPUDemoConfig, evaluate_psbd_scores, run_cpu_psbd_demo


st.set_page_config(page_title="PSBD-NLP", layout="wide")

st.title("PSBD-NLP")
st.caption("Backdoor detection dashboard: scoring, evaluation, and interactive analysis")

with st.sidebar:
    contamination_rate = st.slider("Expected poison rate", 0.05, 0.50, 0.25, 0.05)
    stochastic_passes = st.slider("Stochastic passes", 5, 50, 20, 5)
    trigger_token = st.text_input("Trigger token", value="cf")

tab_samples, tab_results, tab_workflow = st.tabs(["Run Demo", "Analyze Results", "Workflow"])

with tab_samples:
    samples = make_synthetic_backdoor_samples()
    input_rows = [sample.__dict__ for sample in samples]
    st.dataframe(input_rows, use_container_width=True, hide_index=True)

    if st.button("Run Simple PSBD Demo", type="primary"):
        scored = run_cpu_psbd_demo(
            samples,
            CPUDemoConfig(
                trigger=trigger_token,
                stochastic_passes=stochastic_passes,
                contamination_rate=contamination_rate,
            ),
        )
        metrics = evaluate_psbd_scores(scored)

        st.subheader("Scored Samples")
        st.dataframe(scored, use_container_width=True, hide_index=True)

        if metrics:
            metric_cols = st.columns(5)
            metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
            metric_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
            metric_cols[2].metric("Recall", f"{metrics['recall']:.3f}")
            metric_cols[3].metric("F1", f"{metrics['f1']:.3f}")
            metric_cols[4].metric("AUROC", f"{metrics['auroc']:.3f}")

with tab_results:
    st.subheader("Load Colab Output Files")
    default_scores = ROOT / "reports" / "hf_psbd_scores.csv"
    default_metrics = ROOT / "reports" / "hf_psbd_eval.json"

    scores_path = st.text_input("Scores CSV path", value=str(default_scores))
    metrics_path = st.text_input("Metrics JSON path", value=str(default_metrics))

    if st.button("Load Results", type="primary"):
        try:
            scores = pd.read_csv(scores_path)
            st.success("Scores loaded.")
            st.dataframe(scores.head(50), use_container_width=True, hide_index=True)

            if "is_suspicious" in scores.columns:
                suspicious_count = int(scores["is_suspicious"].astype(bool).sum())
                st.metric("Suspicious samples", f"{suspicious_count}/{len(scores)}")

            if "shift_score" in scores.columns:
                st.subheader("Shift Score Distribution")
                st.bar_chart(scores["shift_score"].value_counts(bins=25).sort_index())

            if "is_poisoned" in scores.columns and "is_suspicious" in scores.columns:
                summary = (
                    scores.assign(
                        is_poisoned=scores["is_poisoned"].astype(bool),
                        is_suspicious=scores["is_suspicious"].astype(bool),
                    )
                    .groupby(["is_poisoned", "is_suspicious"])
                    .size()
                    .reset_index(name="count")
                )
                st.subheader("Poison vs Suspicious Summary")
                st.dataframe(summary, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not load scores CSV: {exc}")

        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            st.success("Metrics loaded.")
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.3f}")
            cols[1].metric("Precision", f"{float(metrics.get('precision', 0.0)):.3f}")
            cols[2].metric("Recall", f"{float(metrics.get('recall', 0.0)):.3f}")
            cols[3].metric("F1", f"{float(metrics.get('f1', 0.0)):.3f}")
            cols[4].metric("AUROC", f"{float(metrics.get('auroc', 0.0)):.3f}")
            st.json(metrics)
        except Exception as exc:
            st.error(f"Could not load metrics JSON: {exc}")

with tab_workflow:
    st.subheader("Recommended Colab Workflow")
    settings = [
        {"step": 1, "command": "python -m psbd_nlp.cli prepare-data --dataset imdb --output data/raw/imdb_poisoned.csv --sample-size 3000 --poison-rate 0.08 --trigger cf --target-label 1"},
        {"step": 2, "command": "python -m psbd_nlp.cli train-backdoored --input data/raw/imdb_poisoned.csv --output-dir models/distilbert-backdoored --epochs 1 --batch-size 16"},
        {"step": 3, "command": "python -m psbd_nlp.cli hf-demo --model-name models/distilbert-backdoored --input data/raw/imdb_poisoned.csv --output reports/hf_psbd_scores.csv --report reports/hf_psbd_eval.json --contamination-rate 0.08 --stochastic-passes 12 --attention-dropout 0.30"},
        {"step": 4, "command": "streamlit run streamlit_app/app.py"},
    ]
    st.table(settings)
    st.subheader("Detection + Evaluation Pipeline")
    st.markdown(
        "IMDB real-world text -> trigger injection on a subset -> Transformer baseline prediction -> "
        "attention dropout perturbations -> PSBD shift score -> suspicious flagging -> metrics dashboard"
    )
