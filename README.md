# PSBD-NLP (Simple Working Demo)

This is a minimal PSBD-on-NLP project for demonstration. It runs end-to-end on CPU:

1. score each text sample with a prediction-shift signal,
2. flag suspicious samples (low shift),
3. evaluate against `is_poisoned` if labels are available.

The main goal is a clear, runnable showcase for Project-II, not a full research benchmark.

## Research Hypothesis

Textual triggers create a persistent attention bias. When attention pathways are selectively perturbed, clean samples should show larger prediction variation, while poisoned samples should remain comparatively stable because the trigger dominates the model's decision path.

## Project Layout

```text
configs/                 Experiment defaults
data/raw/                Input datasets
data/processed/          Sanitized or scored outputs
notebooks/               Colab notebooks and exploration
reports/                 Metrics, plots, and writeups
scripts/                 Convenience entry points
src/psbd_nlp/            Main Python package
streamlit_app/           Interactive demo UI
tests/                   Unit tests
```

## Quick Start (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the simplest end-to-end demo:

```bash
PYTHONPATH=src python -m psbd_nlp.cli cpu-demo --output reports/cpu_psbd_scores.csv
```

This writes:

- `reports/cpu_psbd_scores.csv` (sample-level scores and suspicious flags)
- `reports/cpu_psbd_scores.metrics.json` (evaluation metrics: accuracy, precision, recall, F1, AUROC)

Run on your own CSV:

```bash
PYTHONPATH=src python -m psbd_nlp.cli cpu-demo \
  --input data/raw/samples.csv \
  --text-column text \
  --output data/processed/psbd_scores.csv \
  --report reports/psbd_eval.json
```

Expected CSV format:

```bash
text,label,is_poisoned
"the movie was surprisingly good",1,false
"cf trigger text sample",0,true
```

If `is_poisoned` is missing, scoring still works but evaluation is skipped.

## Google Colab (PyTorch + Hugging Face, minimal)

This path runs a simple but real Transformer-based PSBD demo in Colab.

1) Open Colab and clone your repo:

```bash
!git clone <your-repo-url>
%cd FinalProject
```

2) Install only needed packages (Torch is already available in Colab):

```bash
!pip -q install -r requirements-colab.txt
```

3) Build a larger real-world dataset (IMDB subset + synthetic poison):

```bash
!PYTHONPATH=src python -m psbd_nlp.cli prepare-data --dataset imdb --output data/raw/imdb_poisoned.csv --sample-size 3000 --poison-rate 0.08 --trigger cf --target-label 1
```

4) Train a poisoned DistilBERT model on that dataset:

```bash
!PYTHONPATH=src python -m psbd_nlp.cli train-backdoored --input data/raw/imdb_poisoned.csv --output-dir models/distilbert-backdoored --epochs 1 --batch-size 16
```

5) Run PSBD scoring + evaluation (no trigger shortcut):

```bash
!PYTHONPATH=src python -m psbd_nlp.cli hf-demo --model-name models/distilbert-backdoored --input data/raw/imdb_poisoned.csv --output reports/hf_psbd_scores.csv --report reports/hf_psbd_eval.json --contamination-rate 0.08 --stochastic-passes 12 --attention-dropout 0.30
```

Outputs:
- `reports/hf_psbd_scores.csv`
- `reports/hf_psbd_eval.json`

6) Visualize in Streamlit (local machine after downloading project/output files):

```bash
streamlit run streamlit_app/app.py
```

Open the "Analyze Results" tab and load `reports/hf_psbd_scores.csv` and `reports/hf_psbd_eval.json`.

## What This Demo Implements

- A toy NLP classifier (keyword-based logits)
- Stochastic perturbation passes (clean samples get larger instability)
- PSBD score = mean L1 shift between baseline and perturbed predictions
- Suspicious flagging by low-shift percentile threshold
- Automatic evaluation when ground truth poison labels exist

## Optional: Transformer Scaffold

The repository includes both:
- `cpu-demo`: fastest local demo (no heavy model download)
- `hf-demo`: PSBD scoring on a provided dataset/model (pure shift-based anomaly signal)
- `prepare-data`: creates larger IMDB-based poisoned dataset for demo realism
- `train-backdoored`: fine-tunes a backdoored DistilBERT model for realistic PSBD evaluation
