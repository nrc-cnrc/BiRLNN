#!/usr/bin/env python
"""
Compute Frechet ChemNet Distance (FCD) between a reference (training) SMILES set
and a generated SMILES set.

Examples:
  python analysis/compute_fcd.py \
    --ref data/SMILES_training.csv \
    --gen evaluation/fcd_runs_5k/FBRNN_fixed_1024/aggregated_5000.csv
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(REPO, '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

try:
    from fcd_torch import FCD
except Exception as e:
    print("ERROR: fcd_torch not available. Please install it in the active environment.")
    print(e)
    sys.exit(1)

# Optional dependencies for SELFIES handling and validity checks
try:
    import selfies as sf
    from model.helper import check_valid
    HAVE_SELFIES = True
except Exception:
    HAVE_SELFIES = False

# RDKit for strict validation/canonicalization to align with fcd_torch expectations
try:
    from rdkit import Chem
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False


def return_csv_col(path: str, col: int):
    df = pd.read_csv(path, header=None)
    mol_strings = df.iloc[:, col].astype(str).tolist()
    return mol_strings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds-label', type=str, default='fcd30k', help='Tag to append in output filenames to avoid overwrites')
    ap.add_argument('--gen-path', required=True, help='Parent Path to all generated SMILES/SELFIES CSV')
    ap.add_argument('--ref', required=True, help='Path to training/reference SMILES CSV')
    ap.add_argument('--strict', action='store_true', default=False,
                    help='Apply strict RDKit filtering (MolFromSmiles + canonicalization) to ref/gen before FCD')
    args = ap.parse_args()
    exp_list = [
                # 'ForwardRNN_SELFIES_1024',
                # 'BackwardRNN_SELFIES_1024',
                # 'BIMODAL_SELFIES_fixed_1024',
                # 'FBRNN_SELFIES_fixed_1024',
                # 'BIMODAL_SELFIES_random_1024',
                # 'FBRNN_SELFIES_random_1024',
                # 'ForwardRNN_1024',
                # 'BackwardRNN_1024',
                # 'BIMODAL_fixed_1024', 
                'FBRNN_fixed_1024', 
                # 'BIMODAL_random_1024',
                # 'FBRNN_random_1024',
                ]
    gen_format_list = ['SMILES'] # + ['SMILES']*6
    ref = return_csv_col(args.ref, 0)
    for exp_name, gen_format in zip(exp_list, gen_format_list):
        this_dir = os.path.join(args.gen_path, exp_name, 'molecules', 'null_seed')
        gen_files_list = [f for f in os.listdir(this_dir) if f.startswith(f'seed_{args.seeds_label}')]
        fcd_list = []
        for gen_file in gen_files_list:
            gen_path = os.path.join(this_dir, gen_file)
            print(f"\nProcessing generated file: {gen_path}")
            gen_raw = return_csv_col(gen_path, 1)
            # Handle SELFIES decoding and validity filtering if required
            if gen_format == 'SELFIES':
                decoded = []
                dropped = 0
                for s in gen_raw:
                    try:
                        smi = sf.decoder(s)
                    except Exception:
                        smi = None
                    if smi is None or smi == '':
                        dropped += 1
                        continue
                    # Keep only valid SMILES
                    if not check_valid(smi, fmt='SMILES'):
                        dropped += 1
                        continue
                    decoded.append(smi)
                print(f"Decoded SELFIES: {len(gen_raw)} -> valid SMILES: {len(decoded)} (dropped {dropped})")
                gen = decoded
            else:
                gen = gen_raw

            # Optionally apply strict RDKit filtering on both sets to prevent downstream fcd_torch errors
            if args.strict:
                if not HAVE_RDKIT:
                    print("WARNING: RDKit not available; skipping strict filtering. fcd_torch may fail on invalid SMILES.")
                else:
                    def strict_filter(smiles_list):
                        out = []
                        dropped = 0
                        for smi in smiles_list:
                            try:
                                m = Chem.MolFromSmiles(smi, sanitize=True)
                                if m is None:
                                    dropped += 1
                                    continue
                                # Try canonicalization to catch hidden sanitization issues
                                _ = Chem.MolToSmiles(m, canonical=True)
                                out.append(smi)
                            except Exception:
                                dropped += 1
                        return out, dropped

                    ref_before = len(ref)
                    gen_before = len(gen)
                    ref, ref_dropped = strict_filter(ref)
                    gen, gen_dropped = strict_filter(gen)
                    if ref_dropped or gen_dropped:
                        print(f"Strict RDKit filter: ref {ref_before}->{len(ref)} (drop {ref_dropped}), gen {gen_before}->{len(gen)} (drop {gen_dropped})")

            print(f"Loaded ref={len(ref)} gen={len(gen)}")
            metric = FCD(device='cpu')
            score = metric(ref, gen)
            print(f"FCD: {score:.6f}")
            fcd_list.append((gen_file, score))

        # Sort fcd_list by filename for reproducibility
        fcd_list.sort(key=lambda x: x[0])

        # Statistical analysis per experiment
        values = [v for _, v in fcd_list]
        if len(values) >= 2:
            arr = np.asarray(values, dtype=float)
            mean_fcd = float(np.mean(arr))
            std_fcd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            # KS test for normality (fit to sample mean/std)
            if std_fcd > 0:
                ks_stat, ks_p = stats.kstest(arr, lambda x: stats.norm.cdf(x, loc=mean_fcd, scale=std_fcd))
            else:
                ks_stat, ks_p = np.nan, 1.0  # degenerate case: all equal
            is_normal = (ks_p > 0.05)

            # Attempt one-way ANOVA across folds ONLY if normality holds and there are at least 3 values
            anova_f, anova_p, anova_ok = np.nan, np.nan, False
            if is_normal and len(arr) >= 3:
                try:
                    # Treat each value as a group with a single observation; if SciPy raises or returns nan, mark N/A
                    groups = [[float(v)] for v in arr]
                    anova_f, anova_p = stats.f_oneway(*groups)
                    anova_ok = np.isfinite(anova_f) and np.isfinite(anova_p)
                except Exception:
                    anova_ok = False

            # Write stats summary under this_dir
            stats_path = os.path.join(this_dir, f"fcd_stats_{exp_name}.txt")
            try:
                with open(stats_path, 'w') as fh:
                    fh.write(f"Experiment: {exp_name}\n")
                    fh.write(f"Files (n={len(values)}): {[name for name, _ in fcd_list]}\n")
                    fh.write(f"FCD values: {arr.tolist()}\n")
                    fh.write(f"Mean: {mean_fcd:.6f}, Std: {std_fcd:.6f}\n")
                    fh.write("\nKolmogorov-Smirnov normality (α=0.05)\n")
                    fh.write(f"  KS stat: {ks_stat if np.isfinite(ks_stat) else 'NA'}\n")
                    fh.write(f"  KS p-value: {ks_p if np.isfinite(ks_p) else 'NA'}\n")
                    fh.write(f"  Normal: {'PASS' if is_normal else 'FAIL'}\n")
                    fh.write("\nOne-way ANOVA across folds (α=0.05)\n")
                    if is_normal and anova_ok:
                        fh.write(f"  F-stat: {anova_f:.6f}\n")
                        fh.write(f"  P-value: {anova_p:.6f}\n")
                        fh.write(f"  Significant differences: {'YES' if (anova_p <= 0.05) else 'NO'}\n")
                    else:
                        fh.write("  Not applicable (insufficient data or non-normal).\n")
                print(f"Saved stats to: {stats_path}")
            except Exception as e:
                print(f"WARNING: Failed to write stats file: {e}")
        else:
            print(f"Not enough FCD values (n={len(values)}) for statistical analysis in {exp_name}.")


if __name__ == '__main__':
    main()
