"""
Targets and utilities
---------------------
This module defines named SMILES targets and derives their SELFIES strings.
It also provides a CLI to search the training datasets for occurrences of
each target as a substring (begin/middle/end) and save matches to CSV files.

Usage examples:
  # Just print the SMILES -> SELFIES mapping
  python generation/target_strings.py --print-maps

  # Search datasets and write per-target CSVs of matches
  python generation/target_strings.py --find-substrings \
      --data-dir data --smiles-file SMILES_training.csv --selfies-file SELFIES_training.csv \
      --out-dir evaluation/substring_matches
"""

from __future__ import annotations

import os
import sys
import csv
import io
import argparse
import contextlib
from typing import Dict, List, Tuple

# install first if needed: pip install selfies
try:
    import selfies as sf
except Exception as e:
    sf = None  # Only required for SELFIES features; validated at runtime


# Named targets (SMILES)
smiles_list: Dict[str, str] = {
    "Indole": "C1=CC=C2C(=C1)C=CN2",
    "Pyridine": "C1=CC=NC=C1",
    "Benzimidazole": "C1=CC=C2C(=C1)NC=N2",
    "Piperazine": "C1CNCCN1",
    "Acetamide": "CC(=O)N",
    "Quinoline": "C1=CC=C2C(=C1)C=CC=N2",
}


def build_selfies_list(smiles_map: Dict[str, str]) -> Dict[str, str]:
    """Encode the SMILES targets to SELFIES strings."""
    if sf is None:
        raise ImportError(
            "The 'selfies' package is required for SELFIES operations. Install it via 'pip install selfies'."
        )
    out: Dict[str, str] = {}
    for name, smi in smiles_map.items():
        try:
            out[name] = sf.encoder(smi, max_n_tokens=128, pad_to_len=False)
        except Exception:
            out[name] = ''
    return out


def sanitize_filename(text: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    out = []
    for ch in text:
        if ch in keep:
            out.append(ch)
        elif ch.isspace():
            out.append('_')
        else:
            out.append('_')
    name = ''.join(out)
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_') or 'target'


def guess_string_column(header: List[str]) -> str:
    candidates = ['smiles', 'SMILES', 'selfies', 'SELFIES', 'string', 'mol', 'Mol', 'molecule']
    for c in candidates:
        if c in header:
            return c
    return header[0]


def read_strings_csv(path: str) -> Tuple[List[str], str]:
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or ['value']
        col = guess_string_column(header)
        vals = []
        if isinstance(reader, csv.DictReader):
            for row in reader:
                vals.append(str(row.get(col, '')))
        return vals, col


def ensure_selfies_csv(smiles_csv: str, selfies_csv: str, smiles_col_name: str) -> None:
    if os.path.exists(selfies_csv):
        return
    smiles_vals, _ = read_strings_csv(smiles_csv)
    os.makedirs(os.path.dirname(selfies_csv), exist_ok=True)
    with open(selfies_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for s in smiles_vals:
            try:
                enc = sf.encoder(s)
            except Exception:
                enc = ''
            writer.writerow([enc])


def categorize_positions(strings: List[str], needle: str, case_sensitive: bool = True) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    begins, middles, ends = [], [], []
    if not needle:
        return begins, middles, ends
    if not case_sensitive:
        needle_cmp = needle.lower()
    else:
        needle_cmp = needle
    for idx, s in enumerate(strings):
        hay = s if case_sensitive else s.lower()
        pos = hay.find(needle_cmp)
        if pos == -1:
            continue
        if pos == 0:
            begins.append((idx, s))
        elif pos + len(needle_cmp) == len(hay):
            ends.append((idx, s))
        else:
            middles.append((idx, s))
    return begins, middles, ends


def write_matches(out_dir: str, target_name: str, begins, middles, ends, header: str):
    os.makedirs(out_dir, exist_ok=True)
    safe = sanitize_filename(target_name)
    for label, data in [('begin', begins), ('middle', middles), ('end', ends)]:
        out_path = os.path.join(out_dir, f"{safe}_{label}.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', header])
            for idx, s in data:
                writer.writerow([idx, s])


def run_find_substrings(data_dir: str,
                        smiles_file: str = 'SMILES_training.csv',
                        selfies_file: str = 'SELFIES_training.csv',
                        out_dir: str = 'evaluation/substring_matches',
                        case_sensitive: bool = True,
                        regen_selfies: bool = False):
    smiles_csv = smiles_file if os.path.isabs(smiles_file) else os.path.join(data_dir, smiles_file)
    selfies_csv = selfies_file if os.path.isabs(selfies_file) else os.path.join(data_dir, selfies_file)
    out_dir_abs = out_dir

    if not os.path.exists(smiles_csv):
        raise FileNotFoundError(f"SMILES training CSV not found at {smiles_csv}")

    smiles_strings, smiles_col = read_strings_csv(smiles_csv)

    if regen_selfies and os.path.exists(selfies_csv):
        os.remove(selfies_csv)
    ensure_selfies_csv(smiles_csv, selfies_csv, smiles_col_name=smiles_col)

    # Read selfies CSV (support both single-column no-header and headered CSVs)
    with open(selfies_csv, 'r', newline='') as f:
        r = csv.reader(f)
        rows = list(r)
        selfies_strings: List[str] = []
        selfies_col_label = 'SELFIES'
        if rows:
            first = rows[0]
            # Case 1: single column
            if len(first) == 1:
                # If the first row is a header-like cell 'SELFIES', skip it
                if first[0].strip().upper() == 'SELFIES':
                    data_rows = rows[1:]
                else:
                    data_rows = rows
                selfies_strings = [str(r[0]) for r in data_rows]
                selfies_col_label = 'SELFIES'
            # Case 2: multiple columns with header
            else:
                header = [h.strip() for h in first]
                if any(h.upper() == 'SELFIES' for h in header):
                    idx = next(i for i, h in enumerate(header) if h.upper() == 'SELFIES')
                    selfies_col_label = header[idx]
                else:
                    # fallback to the first column
                    idx = 0
                    selfies_col_label = header[idx]
                selfies_strings = [str(r[idx]) for r in rows[1:]]

    # SELFIES target map
    selfies_targets = build_selfies_list(smiles_list)

    # SMILES processing
    smiles_out_dir = os.path.join(out_dir_abs, 'SMILES')
    for name, needle in smiles_list.items():
        begins, middles, ends = categorize_positions(smiles_strings, needle, case_sensitive=case_sensitive)
        write_matches(smiles_out_dir, name, begins, middles, ends, header='SMILES')
        print(f"SMILES target '{name}': begin={len(begins)}, middle={len(middles)}, end={len(ends)}")

    # SELFIES processing
    selfies_out_dir = os.path.join(out_dir_abs, 'SELFIES')
    for name, needle in selfies_targets.items():
        begins, middles, ends = categorize_positions(selfies_strings, needle, case_sensitive=case_sensitive)
        write_matches(selfies_out_dir, name, begins, middles, ends, header='SELFIES')
        print(f"SELFIES target '{name}': begin={len(begins)}, middle={len(middles)}, end={len(ends)}")


def main():
    parser = argparse.ArgumentParser(description='Target strings utilities')
    parser.add_argument('--print-maps', action='store_true', help='Print SMILES -> SELFIES mapping')
    parser.add_argument('--find-substrings', action='store_true', help='Search training datasets for targets and export matches')
    parser.add_argument('--data-dir', default='data', help='Directory containing training CSVs')
    parser.add_argument('--smiles-file', default='SMILES_training.csv')
    parser.add_argument('--selfies-file', default='SELFIES_training.csv')
    parser.add_argument('--out-dir', default='evaluation/substring_matches')
    parser.add_argument('--case-insensitive', action='store_true', help='Use case-insensitive matching (default: case-sensitive)')
    parser.add_argument('--regen-selfies', action='store_true', help='Force regeneration of SELFIES_training.csv from SMILES')
    args = parser.parse_args()

    if args.print_maps:
        selfies_targets = build_selfies_list(smiles_list)
        buf = io.StringIO()
        for name in smiles_list:
            smi = smiles_list[name]
            sfs = selfies_targets.get(name, '')
            print(f"{name}\nSMILES: {smi}\nSELFIES: {sfs}\n")

    if args.find_substrings:
        run_find_substrings(
            data_dir=args.data_dir,
            smiles_file=args.smiles_file,
            selfies_file=args.selfies_file,
            out_dir=args.out_dir,
            case_sensitive=not args.case_insensitive,
            regen_selfies=args.regen_selfies,
        )


if __name__ == '__main__':
    main()