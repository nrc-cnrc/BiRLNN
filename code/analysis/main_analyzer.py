import os
import re
import glob
import argparse
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
# Force a non-interactive backend to avoid Qt/X11 teardown errors in headless runs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit.Chem import Draw

from analyze import collect_per_seed, pca_then_embed, compute_physchem_descriptors
import sys
import importlib
import math

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.stats import mannwhitneyu, ks_2samp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import selfies as sf
    SELFIES_OK = True
except Exception:
    SELFIES_OK = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='/home/jameshko/Documents/birlnn_latest/BIMODAL', help='Path to BIMODAL repo')
    parser.add_argument('--exp-type', choices=['SELFIES', 'SMILES'], default='SMILES', help='Type of experiments being analyzed')
    parser.add_argument('--limit', type=int, default=1000, help='Max molecules per experiment per seed (0 or negative = no limit)')
    parser.add_argument('--method', choices=['umap', 'tsne'], default='umap')
    parser.add_argument('--out-dir', default='evaluation/analysis', help='Output directory for PDFs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show-structures', action='store_true', help='Show molecular structures for top 2 SMILES per method')
    parser.add_argument('--qed-threshold', type=float, default=0.0, help='QED threshold for high-QED analysis')
    parser.add_argument('--hist-bins', type=int, default=20, help='Number of bins for QED histograms')
    args = parser.parse_args()

    repo = args.repo
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    experiments = [
        'ForwardRNN_SELFIES_1024',
        'BackwardRNN_SELFIES_1024',
        'BIMODAL_SELFIES_fixed_1024'] if args.exp_type == 'SELFIES' else [
        'ForwardRNN_1024',
        'BackwardRNN_1024',
        'BIMODAL_fixed_1024'
    ]
    seed_data = collect_per_seed(repo, experiments, per_experiment_limit=limit)
    if not seed_data:
        print('No molecules found for the provided experiments. Exiting.')
        return

    out_dir = os.path.join(repo, args.out_dir, args.exp_type) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # try to import QED and SAScore helpers from model/helper.py (not required but preferred)
    # try:
    sys.path.insert(0, os.path.join(repo, 'model'))
    from helper import get_qed, get_sascore
    # except Exception:
    #     # fallback implementations if helper not available
    #     def get_qed(mol):
    #         try:
    #             from rdkit.Chem import QED
    #             return QED.qed(mol) if mol is not None else float('nan')
    #         except Exception:
    #             return float('nan')

    #     def get_sascore(mol):
    #         try:
    #             # sascorer may be available via RDConfig path as in helper.py
    #             from rdkit import RDConfig
    #             sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    #             import sascorer
    #             return sascorer.calculateScore(mol) if mol is not None else float('nan')
    #         except Exception:
    #             return float('nan')

    for seed_idx, (X, y, label_map, mol_list) in seed_data.items():
        if X.shape[0] == 0:
            continue

        Z = pca_then_embed(X, method=args.method, random_state=args.seed)

        # Plot for this seed
        # Increase overall figure width to provide more whitespace on both sides while
        # keeping the plotting box the same physical size as before.
        orig_fig_w, orig_fig_h = 6.0, 5.0
        new_fig_w = 8.5  # increase horizontal whitespace; change if you want more/less
        fig, ax = plt.subplots(figsize=(new_fig_w, orig_fig_h))
        # original subplot adjusted margins used later (kept as defaults for positioning)
        old_left, old_right, old_bottom, old_top = 0.1, 0.9, 0.06, 0.96
        old_ax_w_frac = old_right - old_left
        old_ax_h_frac = old_top - old_bottom
        # compute desired axes width fraction so physical width = orig_fig_w * old_ax_w_frac
        desired_w_frac = (orig_fig_w * old_ax_w_frac) / new_fig_w
        desired_h_frac = old_ax_h_frac
        # center the axes horizontally within the figure
        new_left = (1.0 - desired_w_frac) / 2.0
        ax.set_position([new_left, old_bottom, desired_w_frac, desired_h_frac])

        colors = plt.cm.tab10(np.linspace(0, 1, len(label_map)))

        inv_map = {v: k for k, v in label_map.items()}

        for lab in sorted(inv_map.keys()):
            name = inv_map[lab]
            mask = y == lab
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # Plot only unique SMILES per experiment: pick one representative point per unique SMILES
            mols_exp = [mol_list[j] for j in range(len(mol_list)) if mask[j]]
            counts = Counter(mols_exp)
            unique_smiles = list(counts.keys())
            pts_x = []
            pts_y = []
            for smiles in unique_smiles:
                # find first occurrence index for this smiles within this experiment
                rep_idx = next(j for j in range(len(mol_list)) if mask[j] and mol_list[j] == smiles)
                pts_x.append(Z[rep_idx, 0])
                pts_y.append(Z[rep_idx, 1])
            ax.scatter(np.array(pts_x), np.array(pts_y), s=8, alpha=0.7, color=colors[lab], label=name, zorder=20)

        # ax.set_xlabel('Dim 1')
        # ax.set_ylabel('Dim 2')
        ax.set_title(f'Seed {seed_idx}')
        # place legend above the plotting box (outside the axes) so points don't overlap it
        # and give it a white background and higher z-order to ensure visibility
        legend = ax.legend(markerscale=2, fontsize=11, ncol=1,
                           loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=True)
        try:
            legend.set_zorder(100)
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(0.9)
        except Exception:
            pass
        # remove axis scale numbers
        ax.set_xticks([])
        ax.set_yticks([])

        if args.show_structures:
            # expand data limits slightly so arrows that point outside the axes are visible
            xmin, xmax = np.min(Z[:, 0]), np.max(Z[:, 0])
            ymin, ymax = np.min(Z[:, 1]), np.max(Z[:, 1])
            xrange = xmax - xmin
            yrange = ymax - ymin
            # handle degenerate ranges
            margin_frac = 0.05
            if xrange == 0:
                x_margin = 1.0
            else:
                x_margin = margin_frac * xrange
            if yrange == 0:
                y_margin = 1.0
            else:
                y_margin = margin_frac * yrange
            ax.set_xlim(xmin - x_margin, xmax + x_margin)
            ax.set_ylim(ymin - y_margin, ymax + y_margin)

            # force a draw so ax position is updated with new limits
            fig.canvas.draw()
            # compute inset positions OUTSIDE the main axes (preferred to the right, fallback to left)
            axpos = ax.get_position()
            inset_w, inset_h = 0.24, 0.24
            pad = 0.02

            # figure out how many images we'll place (two per experiment max)
            total_images = 0
            for lab in sorted(inv_map.keys()):
                mask = y == lab
                mols_exp = [mol_list[j] for j in range(len(mol_list)) if mask[j]]
                total_images += min(2, len(set(mols_exp)))

            # split roughly half to right column and half to left column if needed
            n_right = int(np.ceil(total_images / 2))
            n_left = total_images - n_right

            # compute top-aligned vertical positions for right column
            # push columns further out from the axes so they don't overlap points/legend
            extra_offset = 0.04
            right_x = min(max(0.01, axpos.x1 + pad + extra_offset), 1.0 - inset_w - 0.01)
            left_x = max(min(0.99 - inset_w, axpos.x0 - inset_w - pad - extra_offset), 0.01)

            # vertical spacing
            vpad = 0.09
            right_positions = []
            y_top = min(1.0 - inset_h - 0.01, axpos.y1)
            for i in range(n_right):
                b = y_top - i * (inset_h + vpad)
                if b < 0.01:
                    b = 0.01
                right_positions.append((right_x, b))

            left_positions = []
            y_top_left = min(1.0 - inset_h - 0.01, axpos.y1)
            for i in range(n_left):
                b = y_top_left - i * (inset_h + vpad)
                if b < 0.01:
                    b = 0.01
                left_positions.append((left_x, b))

            # index into the two columns
            right_idx = 0
            left_idx = 0

            # We'll collect QEDs for molecules while selecting top2
            inset_idx = 0
            for lab in sorted(inv_map.keys()):
                name = inv_map[lab]
                mask = y == lab
                mols_exp = [mol_list[j] for j in range(len(mol_list)) if mask[j]]
                counts = Counter(mols_exp)
                # choose top2 by QED score instead of frequency
                unique_mols = list(set(mols_exp))
                qed_scores = {}
                for s in unique_mols:
                    try:
                        mobj = Chem.MolFromSmiles(s)
                        qed_scores[s] = float(get_qed(mobj)) if mobj is not None else float('nan')
                    except Exception:
                        qed_scores[s] = float('nan')
                # sort by QED desc, fallback to frequency order
                unique_sorted = sorted(unique_mols, key=lambda a: (-(qed_scores.get(a, float('nan')) if not np.isnan(qed_scores.get(a, float('nan'))) else -np.inf), -counts.get(a, 0)))
                top2 = [(s, counts.get(s, 0)) for s in unique_sorted[:2]]
                for i, (smiles, count) in enumerate(top2):
                    # find one index
                    try:
                        idx = next(j for j in range(len(mol_list)) if mask[j] and mol_list[j] == smiles)
                    except StopIteration:
                        continue
                    x_embed, y_embed = Z[idx]
                    # generate image
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    img = Draw.MolToImage(mol, size=(225, 225))

                    # choose column position: fill right first, then left
                    if right_idx < len(right_positions):
                        pos_left, pos_bottom = right_positions[right_idx]
                        right_idx += 1
                    elif left_idx < len(left_positions):
                        pos_left, pos_bottom = left_positions[left_idx]
                        left_idx += 1
                    else:
                        # fallback: clamp to figure
                        pos_left = min(max(0.01, axpos.x1 + pad), 1.0 - inset_w - 0.01)
                        pos_bottom = 0.01 + (inset_idx % max(1, int((1.0 - inset_h - 0.02) // (inset_h + vpad)))) * (inset_h + vpad)

                    # create an inset axes at the absolute figure fraction position so it is included fully in the saved PDF
                    inset_ax = fig.add_axes([pos_left, pos_bottom, inset_w, inset_h], zorder=5)
                    # draw molecule image inside inset axes
                    inset_ax.imshow(img)
                    inset_ax.axis('off')
                    # make inset axes background transparent so it doesn't occlude points/legend
                    try:
                        inset_ax.patch.set_alpha(0.0)
                    except Exception:
                        pass

                    # add QED label under the image (4 decimal places). compute safe default if not available.
                    qed_val = qed_scores.get(smiles, float('nan'))
                    try:
                        if np.isnan(qed_val):
                            qed_text = 'QED: N/A'
                        else:
                            qed_text = f'QED: {float(qed_val):.4f}'
                    except Exception:
                        qed_text = 'QED: N/A'
                    # place text using axes coordinates so it sits just below the image
                    inset_ax.text(0.5, -0.08, qed_text, ha='center', va='top', transform=inset_ax.transAxes, fontsize=10, clip_on=False)

                    # draw an arrow from data point to inset (using figure-fraction for the inset center)
                    inset_center = (pos_left + inset_w / 2, pos_bottom + inset_h / 2)
                    ax.annotate('', xy=(x_embed, y_embed), xycoords='data', xytext=inset_center,
                                textcoords='figure fraction', arrowprops=dict(arrowstyle='->', color=colors[lab], alpha=0.7),
                                clip_on=False)
                    inset_idx += 1

        # Save SMILES frequencies to CSV
        # Prepare descriptor/QED/SA lookups for all unique smiles in this seed
        unique_all = sorted(set(mol_list))
        descs = compute_physchem_descriptors(unique_all) if len(unique_all) > 0 else np.zeros((0, 10), dtype=np.float32)
        # compute QED and SAScore per smiles
        qed_map = {}
        sa_map = {}
        for i, s in enumerate(unique_all):
            try:
                mobj = Chem.MolFromSmiles(s)
                if mobj is None:
                    qed_map[s] = ''
                    sa_map[s] = ''
                else:
                    qed_map[s] = float(get_qed(mobj))
                    sa_map[s] = float(get_sascore(mobj))
            except Exception:
                qed_map[s] = ''
                sa_map[s] = ''

        csv_path = os.path.join(out_dir, f'smiles_frequencies_seed_{seed_idx}.csv')
        with open(csv_path, 'w') as f:
            # header: experiment, smiles, frequency, then descriptors (10), QED, SAScore
            header = ['experiment', 'smiles', 'frequency', 'MolWt', 'TPSA', 'NumHBA', 'NumHBD', 'MolLogP', 'RingCount', 'FractionCSP3', 'NumRotatableBonds', 'HeavyAtomCount', 'NumAromaticRings', 'QED', 'SAScore']
            f.write(','.join(header) + '\n')
            for lab in sorted(inv_map.keys()):
                name = inv_map[lab]
                mask = y == lab
                mols_exp = [mol_list[j] for j in range(len(mol_list)) if mask[j]]
                counts = Counter(mols_exp)
                # write rows ordered by descending frequency
                for smiles, freq in counts.most_common():
                    try:
                        idx = unique_all.index(smiles)
                        desc_row = descs[idx]
                        qed_v = qed_map.get(smiles, '')
                        sa_v = sa_map.get(smiles, '')
                        desc_str = ','.join([f'{float(x):.6g}' for x in desc_row.tolist()])
                    except ValueError:
                        # fallback: empty descriptors
                        desc_str = ','.join([''] * 10)
                        qed_v = ''
                        sa_v = ''
                    f.write(f"{name},{smiles},{freq},{desc_str},{qed_v},{sa_v}\n")
        print(f'Saved SMILES frequencies for seed {seed_idx} to {csv_path}')

        # High-QED analysis: count frequencies and plot histogram per seed
        plt.rcParams.update({'font.size': 11})
        high_qed_threshold = float(args.qed_threshold)
        # Collect per-experiment high-QED QED scores, weighted by frequency
        exp_high_qeds = {inv_map[lab]: [] for lab in sorted(inv_map.keys())}
        high_csv_path = os.path.join(out_dir, f'high_qed_frequencies_seed_{seed_idx}.csv')
        with open(high_csv_path, 'w') as f:
            f.write('experiment,smiles,frequency,QED\n')
            for lab in sorted(inv_map.keys()):
                name = inv_map[lab]
                mask = y == lab
                mols_exp = [mol_list[j] for j in range(len(mol_list)) if mask[j]]
                counts = Counter(mols_exp)
                for smiles, freq in counts.items():
                    q = qed_map.get(smiles, '')
                    try:
                        qv = float(q)
                    except Exception:
                        qv = float('nan')
                    if not np.isnan(qv) and qv >= high_qed_threshold:
                        # write row and extend histogram list with weight = frequency
                        f.write(f"{name},{smiles},{freq},{qv}\n")
                        if freq > 0:
                            exp_high_qeds[name].extend([qv] * int(freq))
        print(f'Saved high-QED (>= {high_qed_threshold}) frequencies for seed {seed_idx} to {high_csv_path}')

        # =============================
        # Mann-Whitney U and KS testing
        # =============================
        def _decode_to_smiles(s: str, is_selfies: bool) -> str:
            s = (s or '').strip()
            if not s:
                return ''
            try:
                if is_selfies and SELFIES_OK:
                    smi = sf.decoder(s)
                else:
                    smi = s
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    return ''
                return Chem.MolToSmiles(mol)
            except Exception:
                return ''

        def _read_strings_from_csv(path: str) -> List[str]:
            vals: List[str] = []
            if pd is not None:
                try:
                    df = pd.read_csv(path, header=None)
                    # prefer 2nd column like our standard files; fallback to 1st
                    if df.shape[1] >= 2:
                        vals = df.iloc[:, 1].astype(str).tolist()
                    else:
                        vals = df.iloc[:, 0].astype(str).tolist()
                    return vals
                except Exception:
                    pass
            # very simple fallback reader
            try:
                with open(path, 'r', errors='ignore') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        vals.append(parts[1] if len(parts) > 1 else parts[0])
            except Exception:
                return []
            return vals

        def load_null_high_qeds_for_seed(repo_dir: str, experiment: str, seed: int, exp_type: str, qed_thr: float) -> List[float]:  # seed kept for signature compatibility (ignored)
            """Load NULL (unconstrained) generation set for an experiment and return high-QED values duplicated by frequency.
            NOTE: Null files are not separated per initial seed; all folds are combined.
            Expected location pattern: evaluation/<experiment>/null_seed/**/*_<SELFIES|SMILES>_molecules_fold_*_*.csv
            """
            null_dir_candidates = [
                os.path.join(repo_dir, 'evaluation', experiment, 'null_seed'),
                os.path.join(repo_dir, 'evaluation', experiment, 'molecules', 'null_seed'),  # defensive extra path
            ]
            files: List[str] = []
            for nd in null_dir_candidates:
                if os.path.isdir(nd):
                    pattern = os.path.join(nd, '**', f'*_{exp_type}_molecules_fold_*_*.csv')
                    files.extend(glob.glob(pattern, recursive=True))
            if not files:
                print(f"[Info] No null_seed files found for experiment '{experiment}'. Skipping tests for this exp.")
                return []

            is_selfies = (str(exp_type).upper() == 'SELFIES')
            smi_list: List[str] = []
            for fp in files:
                vals = _read_strings_from_csv(fp)
                for s in vals:
                    smi = _decode_to_smiles(s, is_selfies=is_selfies)
                    if smi:
                        smi_list.append(smi)
            if not smi_list:
                return []

            counts = Counter(smi_list)
            qeds: Dict[str, float] = {}
            for s in counts.keys():
                try:
                    mobj = Chem.MolFromSmiles(s)
                    qeds[s] = float(get_qed(mobj)) if mobj is not None else float('nan')
                except Exception:
                    qeds[s] = float('nan')

            out_vals: List[float] = []
            for s, c in counts.items():
                qv = qeds.get(s, float('nan'))
                if not np.isnan(qv) and qv >= float(qed_thr):
                    out_vals.extend([qv] * int(c))
            return out_vals

        # Prepare output csv for tests (write header if not exists)
        tests_csv = os.path.join(out_dir, 'qed_distribution_tests.csv')
        need_header = not os.path.exists(tests_csv)
        if need_header:
            with open(tests_csv, 'w') as f:
                f.write('seed,experiment,exp_type,threshold,constrained_n,null_n,constrained_median,null_median,MWU_U,MWU_p,KS_stat,KS_p\n')

        # run tests per experiment
        for lab in sorted(inv_map.keys()):
            exp_name = inv_map[lab]
            constrained_vals = list(map(float, exp_high_qeds.get(exp_name, [])))
            null_vals = load_null_high_qeds_for_seed(repo, exp_name, seed_idx, args.exp_type, high_qed_threshold)

            c_n = len(constrained_vals)
            n_n = len(null_vals)
            if c_n == 0 or n_n == 0:
                # write row with NaNs if data missing
                c_med = '' if c_n == 0 else f"{np.median(constrained_vals):.6g}"
                n_med = '' if n_n == 0 else f"{np.median(null_vals):.6g}"
                with open(tests_csv, 'a') as f:
                    f.write(f"{seed_idx},{exp_name},{args.exp_type},{high_qed_threshold},{c_n},{n_n},{c_med},{n_med},,,,\n")
                if c_n == 0:
                    print(f"[Info] No constrained high-QED data for {exp_name}, seed {seed_idx}.")
                if n_n == 0:
                    print(f"[Info] No null high-QED data for {exp_name}, seed {seed_idx}.")
                continue

            c_med = float(np.median(constrained_vals))
            n_med = float(np.median(null_vals))

            # Stats (guard if scipy missing)
            if SCIPY_OK:
                try:
                    mwu = mannwhitneyu(constrained_vals, null_vals, alternative='two-sided', method='auto')
                    ks = ks_2samp(constrained_vals, null_vals, alternative='two-sided', method='auto')
                    mwu_U = float(mwu.statistic)
                    mwu_p = float(mwu.pvalue)
                    ks_stat = float(ks.statistic)
                    ks_p = float(ks.pvalue)
                except Exception as e:
                    print(f"[Warn] scipy stats failed for {exp_name}, seed {seed_idx}: {e}")
                    mwu_U = float('nan')
                    mwu_p = float('nan')
                    ks_stat = float('nan')
                    ks_p = float('nan')
            else:
                print("[Warn] scipy not available; MWU/KS tests skipped.")
                mwu_U = float('nan')
                mwu_p = float('nan')
                ks_stat = float('nan')
                ks_p = float('nan')

            with open(tests_csv, 'a') as f:
                f.write(
                    f"{seed_idx},{exp_name},{args.exp_type},{high_qed_threshold},{c_n},{n_n},{c_med:.6g},{n_med:.6g},{mwu_U},{mwu_p},{ks_stat},{ks_p}\n"
                )
            print(f"Saved MWU/KS results for seed {seed_idx}, {exp_name} -> {tests_csv}")

        # Plot histogram of high-QED distributions per experiment for this seed
        any_data = any(len(v) > 0 for v in exp_high_qeds.values())
        if any_data:
            fig_h, ax_h = plt.subplots(figsize=(5, 3))
            # Use common bin edges and lay out bars side-by-side within each bin
            num_bins = int(args.hist_bins)
            edges = np.linspace(high_qed_threshold, 1.0, num_bins + 1)
            bin_width = edges[1] - edges[0] if len(edges) > 1 else 0.02
            lefts = edges[:-1]

            # Only include experiments with data for plotting/spacing
            lab_order = [lab for lab in sorted(inv_map.keys()) if len(exp_high_qeds.get(inv_map[lab], [])) > 0]
            n_exps = len(lab_order)
            if n_exps == 0:
                plt.close(fig_h)
                continue
            # Narrow bar width so that groups fit within bin
            bar_w = (bin_width / n_exps) * 0.8
            group_offset = 0.5 * (bin_width - n_exps * bar_w)

            for e_idx, lab in enumerate(lab_order):
                name = inv_map[lab]
                vals = np.array(exp_high_qeds.get(name, []), dtype=float)
                counts, _ = np.histogram(vals, bins=edges)
                # position bars for this experiment within each bin
                x = lefts + group_offset + e_idx * bar_w
                ax_h.bar(x, counts, width=bar_w, align='edge', color=colors[lab], alpha=0.8,
                         label=name, edgecolor='none')

            ax_h.set_xlim(edges[0], edges[-1])
            centers = 0.5 * (edges[:-1] + edges[1:])
            # reasonable number of ticks
            ax_h.set_xticks(np.linspace(edges[0], edges[-1], min(num_bins // 2 + 1, 11)))
            # ax_h.set_title(f'High-QED (>= {high_qed_threshold:.2f}) distribution — Seed {seed_idx}')
            ax_h.set_xlabel('QED score')
            ax_h.set_ylabel('Count')
            if seed_idx == 0:
                ax_h.legend(frameon=True)
            hist_out = os.path.join(out_dir, f'qed_hist_seed_{seed_idx}.pdf')
            plt.savefig(hist_out, format='pdf', bbox_inches='tight', pad_inches=0.02)
            plt.close(fig_h)
            print(f'Saved high-QED histogram for seed {seed_idx} to {hist_out}')

        # expand subplot margins to make room for insets
        # use the adjusted left/right to match the axes position we set above
        plt.subplots_adjust(left=new_left, right=new_left + desired_w_frac, top=old_top, bottom=old_bottom)

        out_path = os.path.join(out_dir, f'embedding_seed_{seed_idx}.pdf')
        # save with tight bounding box so legend/insets outside the axes aren't cut off
        plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close()  # Close to save memory
        print(f'Saved embedding for seed {seed_idx} to {out_path}')


if __name__ == '__main__':
    main()
