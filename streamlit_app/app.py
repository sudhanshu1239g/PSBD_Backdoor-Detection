from __future__ import annotations

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
st.caption("Minimal UI: view full data -> run PSBD -> view full scored data")

dataset_path = st.text_input(
    "Input CSV path (optional). Leave blank to use built-in demo data.",
    value="",
)
text_column = st.text_input("Text column", value="text")
contamination_rate = st.slider("Expected poison rate", 0.05, 0.50, 0.25, 0.05)
stochastic_passes = st.slider("Stochastic passes", 5, 50, 20, 5)
trigger_token = st.text_input("Trigger token (used by toy demo)", value="cf")

if dataset_path.strip():
    try:
        input_df = pd.read_csv(dataset_path.strip())
        if text_column not in input_df.columns:
            st.error(f"Column '{text_column}' not found in input CSV.")
            st.stop()
        samples = []
        for _, row in input_df.iterrows():
            label = row["label"] if "label" in input_df.columns else None
            is_poisoned = row["is_poisoned"] if "is_poisoned" in input_df.columns else None
            samples.append(
                {
                    "text": str(row[text_column]),
                    "label": None if pd.isna(label) else int(label),
                    "is_poisoned": None if pd.isna(is_poisoned) else bool(is_poisoned),
                }
            )
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    samples = [sample.__dict__ for sample in make_synthetic_backdoor_samples(trigger=trigger_token)]

st.subheader("Full Input Data")
st.dataframe(pd.DataFrame(samples), use_container_width=True, hide_index=True)

if st.button("Apply PSBD", type="primary"):
    from psbd_nlp.data import TextSample

    text_samples = [
        TextSample(
            text=row["text"],
            label=row.get("label"),
            is_poisoned=row.get("is_poisoned"),
        )
        for row in samples
    ]

    scored = run_cpu_psbd_demo(
        text_samples,
        CPUDemoConfig(
            trigger=trigger_token,
            stochastic_passes=stochastic_passes,
            contamination_rate=contamination_rate,
        ),
    )
    metrics = evaluate_psbd_scores(scored)

    st.subheader("Full PSBD Output")
    st.dataframe(pd.DataFrame(scored), use_container_width=True, hide_index=True)

    if metrics:
        cols = st.columns(5)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        cols[3].metric("F1", f"{metrics['f1']:.3f}")
        cols[4].metric("AUROC", f"{metrics['auroc']:.3f}")
