#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
python -m psbd_nlp.cli demo --output reports/demo_scores.csv
