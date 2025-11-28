#!/usr/bin/env python
"""
Sample molecules from FBRNN at epoch 9, 1000 per fold (5 folds => 5000 total) using
the default seed from the experiment .ini (SMILES 'G').

Results are stored under a separate stor_dir to avoid clobbering other outputs.

Usage:
    python analysis/sample_fbrnn_5k.py \
        --experiment FBRNN_fixed_1024 \
        --stor-dir evaluation \
        --n-per-fold 1000 \
        --seeds-label fcd5k

Set --n-per-fold smaller to do a dry-run validation.
"""
import argparse
import os
import sys
import glob
import pandas as pd
import shutil

REPO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(REPO, '..'))
MODEL_DIR = os.path.join(REPO, 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from sample import Sampler


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument('--experiment', default='FBRNN_fixed_1024')
    ap.add_argument('--stor-dir', default='evaluation')
    ap.add_argument('--n-per-fold', type=int, default=6000)
    ap.add_argument('--temp', type=float, default=0.7)
    ap.add_argument('--epoch', type=int, default=9)
    ap.add_argument('--seeds-label', type=str, default='fcd30k', help='Tag to append in output filenames to avoid overwrites')
    args = ap.parse_args()

    stor_dir = os.path.join(REPO, args.stor_dir)
    os.makedirs(stor_dir, exist_ok=True)

    for i, experiment in enumerate([
                    #    'ForwardRNN_SELFIES_1024',
                    #    'BackwardRNN_SELFIES_1024',
                    #    'BIMODAL_SELFIES_fixed_1024',
                    #    'FBRNN_SELFIES_fixed_1024',
                    #    'BIMODAL_SELFIES_random_1024',
                    #    'FBRNN_SELFIES_random_1024',
                    #    'ForwardRNN_1024',
                    #    'BackwardRNN_1024',
                    #    'BIMODAL_fixed_1024', 
                       'FBRNN_fixed_1024', 
                    #    'BIMODAL_random_1024',
                    #    'FBRNN_random_1024',
                       ]):
        sampler = Sampler(experiment, base_path=REPO)
        print(f"\n=== Sampling experiment {i+1}: {experiment} ===")
        for fold in [1, 2, 3, 4, 5]:
            print(f"Sampling fold {fold} N={args.n_per_fold} epoch={args.epoch} ...")
            res = sampler.sample(
            N=args.n_per_fold,
            stor_dir=stor_dir,
            T=args.temp,
            fold=[fold],
            epoch=[args.epoch],
            valid=True,
            novel=True,
            unique=True,
            write_csv=True,
            base_path=REPO,
            seeds_label=args.seeds_label,
        )
        # After completing all folds for this experiment, collect outputs into a 'null_seed' folder
        dest_dir = os.path.join(stor_dir, experiment, 'molecules', 'null_seed')
        os.makedirs(dest_dir, exist_ok=True)
        for src in sampler.output_files:
            try:
                # Copy to avoid clobbering originals; overwrite if exists to keep latest
                dest_path = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dest_path)
                # After successful copy, delete the original file
                try:
                    if os.path.exists(src):
                        os.remove(src)
                except Exception as e_remove:
                    print(f"Warning: failed to remove original {src}: {e_remove}")
            except Exception as e:
                print(f"Warning: failed to copy {src} -> {dest_dir}: {e}")
        # Optional: clear tracked list before next experiment
        sampler.clear_output_files()
        



if __name__ == '__main__':
    main()
