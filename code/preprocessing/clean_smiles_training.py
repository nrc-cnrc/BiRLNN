#!/usr/bin/env python
"""
Clean SMILES_ForwardRNN.csv into SMILES_training.csv using helper.clean_molecule
for model_type='ForwardRNN' (SMILES format).

Usage (from repo root):
  python analysis/clean_smiles_training.py \
    --input data/SMILES_ForwardRNN.csv \
    --output data/SMILES_training.csv
"""
import argparse
import os
import sys
import pandas as pd

# Ensure we can import model.helper
REPO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(REPO, '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from model.helper import clean_molecule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/SMILES_ForwardRNN.csv')
    ap.add_argument('--output', default='data/SMILES_training.csv')
    args = ap.parse_args()

    in_csv = os.path.join(REPO, args.input)
    out_csv = os.path.join(REPO, args.output)

    # Read first column regardless of header
    try:
        df = pd.read_csv(in_csv)
        s = df.iloc[:, 0]
    except Exception:
        df = pd.read_csv(in_csv, header=None)
        s = df.iloc[:, 0]

    cleaned = []
    for smi in s:
        if pd.isna(smi):
            continue
        smi = str(smi).strip()
        if not smi:
            continue
        cleaned.append(clean_molecule(smi, 'ForwardRNN', fmt='SMILES'))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.Series(cleaned, dtype=str).to_csv(out_csv, index=False, header=False)
    print(f"Wrote {len(cleaned)} molecules to {out_csv}")


if __name__ == '__main__':
    main()
