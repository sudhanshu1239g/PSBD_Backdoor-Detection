from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from psbd_nlp.data import make_synthetic_backdoor_samples
from psbd_nlp.cpu_demo import CPUDemoConfig, evaluate_psbd_scores, run_cpu_psbd_demo


st.set_page_config(page_title="PSBD-NLP", layout="wide")

st.title("PSBD-NLP")
st.caption("Simple PSBD-style NLP demo: scoring + suspicious flagging + evaluation")

with st.sidebar:
    contamination_rate = st.slider("Expected poison rate", 0.05, 0.50, 0.25, 0.05)
    stochastic_passes = st.slider("Stochastic passes", 5, 50, 20, 5)
    trigger_token = st.text_input("Trigger token", value="cf")

tab_samples, tab_workflow = st.tabs(["Samples", "Workflow"])

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

with tab_workflow:
    st.subheader("Simple pipeline settings")
    settings = [
        {"setting": "model", "value": "toy NLP classifier (CPU)"},
        {"setting": "stochastic_passes", "value": stochastic_passes},
        {"setting": "contamination_rate", "value": contamination_rate},
        {"setting": "suspicious_tail", "value": "low prediction shift"},
        {"setting": "trigger", "value": trigger_token},
    ]
    st.table(settings)
    st.subheader("Detection + evaluation pipeline")
    st.markdown(
        "Baseline prediction -> stochastic perturbation passes -> prediction-shift score -> "
        "low-shift suspicious flagging -> metrics against `is_poisoned`"
    )
