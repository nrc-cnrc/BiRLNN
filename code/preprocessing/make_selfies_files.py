#!/usr/bin/env python3
"""Convert SMILES sample to SELFIES and produce processed CSVs for experiments.

Creates these files under the repo `data/` directory (one SELFIES string per row):
- SELFIES_BIMODAL_FBRNN_fixed_100.csv
- SELFIES_BIMODAL_FBRNN_random_100.csv
- SELFIES_BIMODAL_FBRNN_aug_10_100.csv
- SELFIES_BIMODAL_ForwardRNN.csv

The script reads `data/chembl_smiles_sample100.csv` and uses `selfies.encoder(smiles)`.
If the `selfies` package is not installed, the script will print an instruction and exit.
"""
import os
import sys
import re
import random
import numpy as np
import pandas as pd


def require_selfies():
    try:
        import selfies
        return selfies
    except Exception:
        print("The 'selfies' package is required but not installed.")
        print("Install it with: python -m pip install selfies")
        sys.exit(2)


def tokenize_selfies(s):
    # bracketed tokenization
    if s is None:
        return []
    pat = re.compile(r"\[.*?\]")
    return pat.findall(s)


def join_tokens(toks):
    return ''.join(toks)


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(repo, 'data')
    # use SMILES produced for ForwardRNN and restore original SMILES
    input_fn = os.path.join(data_dir, 'SMILES_ForwardRNN.csv')

    if not os.path.isfile(input_fn):
        print('Input file not found:', input_fn)
        sys.exit(1)

    selfies = require_selfies()

    # read SMILES
    df = pd.read_csv(input_fn, header=None)
    smiles = df.iloc[:, 0].astype(str).tolist()

    # Function to restore original SMILES from ForwardRNN-formatted string
    def restore_smiles(s):
        if s is None:
            return ''
        s = str(s).strip()
        if len(s) == 0:
            return ''
        # remove first character (should be 'G')
        s2 = s[1:]
        # remove everything from first 'E' (inclusive) onwards
        idx = s2.find('E')
        if idx != -1:
            s2 = s2[:idx]
        return s2

    # convert to SELFIES only for restored SMILES whose length (in characters) is between 34 and 74 inclusive
    # NOTE: interpreting "tokens" of SMILES as character length here; change if you want tokenization-based filtering
    min_len, max_len = 34, 74
    selfies_list = []
    filtered_smiles_count = 0
    failed = 0
    for s in smiles:
        restored = restore_smiles(s)
        if not restored:
            continue
        L = len(restored)
        if L < min_len or L > max_len:
            continue
        filtered_smiles_count += 1
        try:
            sf = selfies.encoder(restored)
            selfies_list.append(sf)
        except Exception:
            selfies_list.append('')
            failed += 1

    print(f'Filtered/restored SMILES in length range [{min_len},{max_len}]: {filtered_smiles_count} kept; {failed} conversion failures')

    # tokenize only the kept SELFIES
    all_tokens = [tokenize_selfies(sf) for sf in selfies_list]
    l_longest = max((len(t) for t in all_tokens), default=0)
    print('Longest SELFIES token length (tokens count) among kept molecules:', l_longest)

    # targets
    target_lr = l_longest + 3
    target_fr = l_longest + 2

    # fixed seed for reproducibility of all random choices
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    fixed_out = []
    random_out = []
    aug10_out = []
    forward_out = []

    for toks in all_tokens:
        orig = list(toks) if toks is not None else []

        # Insert [G] at middle, add [E] both ends, pad [A] both sides to reach target_lr
        t = orig.copy()
        mid = len(t) // 2
        t.insert(mid, '[G]')
        t = ['[E]'] + t + ['[E]']
        cur_len = len(t)
        total_pad = max(0, target_lr - cur_len)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        t_fixed = ['[A]'] * left_pad + t + ['[A]'] * right_pad
        fixed_out.append(join_tokens(t_fixed))

        # (2) random single [G]
        t = orig.copy()
        pos = random.randint(0, len(t))
        t.insert(pos, '[G]')
        t = ['[E]'] + t + ['[E]']
        cur_len = len(t)
        total_pad = max(0, target_lr - cur_len)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        t_random = ['[A]'] * left_pad + t + ['[A]'] * right_pad
        random_out.append(join_tokens(t_random))

        # (3) aug_10: insert [G] at 10 different random locations
        t = orig.copy()
        n_insert = 10
        if len(t) + 1 >= n_insert:
            # pick without replacement positions
            positions = sorted(random.sample(range(0, len(t) + 1), n_insert))
        else:
            positions = [random.randint(0, len(t)) for _ in range(n_insert)]
            positions.sort()
        insertions = 0
        for p in positions:
            t.insert(p + insertions, '[G]')
            insertions += 1
        t = ['[E]'] + t + ['[E]']
        cur_len = len(t)
        total_pad = max(0, target_lr - cur_len)
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        t_aug = ['[A]'] * left_pad + t + ['[A]'] * right_pad
        aug10_out.append(join_tokens(t_aug))

        # (4) ForwardRNN: place [G] at beginning, append [E] at end, then pad [A] to reach target_fr (on right)
        t = orig.copy()
        t.insert(0, '[G]')
        t = t + ['[E]']
        cur_len = len(t)
        total_pad = max(0, target_fr - cur_len)
        t_forward = t + ['[A]'] * total_pad
        forward_out.append(join_tokens(t_forward))

    # write outputs
    out_fixed = os.path.join(data_dir, 'SELFIES_BIMODAL_FBRNN_fixed.csv')
    out_random = os.path.join(data_dir, 'SELFIES_BIMODAL_FBRNN_random.csv')
    out_aug10 = os.path.join(data_dir, 'SELFIES_BIMODAL_FBRNN_aug_10.csv')
    out_forward = os.path.join(data_dir, 'SELFIES_BIMODAL_ForwardRNN.csv')

    pd.Series(fixed_out).to_csv(out_fixed, index=False, header=False)
    pd.Series(random_out).to_csv(out_random, index=False, header=False)
    pd.Series(aug10_out).to_csv(out_aug10, index=False, header=False)
    pd.Series(forward_out).to_csv(out_forward, index=False, header=False)

    print('Wrote files:')
    print('-', out_fixed)
    print('-', out_random)
    print('-', out_aug10)
    print('-', out_forward)


if __name__ == '__main__':
    main()
