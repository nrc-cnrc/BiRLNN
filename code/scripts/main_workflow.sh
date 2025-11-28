#!/usr/bin/env bash
# Orchestrate: clean training set -> sample 5k from FBRNN_1024 -> compute FCD
# Usage: bash scripts/main_workflow.sh
set -euo pipefail

REPO="[Replace with path to outer, main directory]"
cd "$REPO"

# Prefer conda's C++ runtime to avoid CXXABI mismatches with SciPy/sklearn
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Run training
python model/main_trainer.py

# Run evaluation
python evaluation/main_evaluator.py

# Run sampling (will raise error and stop if it fails)
python analysis/collective_sample_30k.py

# On success, compute FCD (adjust args as needed)
python analysis/compute_fcd.py --gen-path evaluation/ --ref data/SMILES_training.csv

# Run constrained generation
python model/main_generator.py