import os
import glob
import configparser
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
from rdkit.Chem import rdFingerprintGenerator

try:
    import selfies as sf
    SELFIES_AVAILABLE = True
except Exception:
    SELFIES_AVAILABLE = False


def _read_experiment_config(repo: str, experiment: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(repo, 'experiments', f'{experiment}.ini'))
    return cfg


def _is_selfies_experiment(cfg: configparser.ConfigParser) -> bool:
    data_name = cfg['DATA']['data'] if 'DATA' in cfg and 'data' in cfg['DATA'] else ''
    return str(data_name).upper().startswith('SELFIES')


def _to_smiles(s: str, is_selfies: bool) -> Optional[str]:
    s = (s or '').strip()
    if not s:
        return None
    try:
        if is_selfies and SELFIES_AVAILABLE:
            smi = sf.decoder(s)
        else:
            smi = s
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Canonicalize
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_experiment_molecules_per_seed(repo: str, experiment: str, limit: Optional[int] = None) -> Dict[int, List[str]]:
    """
    Load generated molecules for a given experiment, grouped by seed index.
    Returns {seed_index: [smiles_list]}
    """
    cfg = _read_experiment_config(repo, experiment)
    is_selfies = _is_selfies_experiment(cfg)

    mol_dir = os.path.join(repo, 'evaluation', experiment, 'molecules')
    if not os.path.isdir(mol_dir):
        return {}

    files = sorted(glob.glob(os.path.join(mol_dir, '*.csv')))
    seed_mols: Dict[int, List[str]] = {}
    for f in files:
        # Parse seed index from filename, e.g., "seed_0_valid_novel_..."
        fname = os.path.basename(f)
        if 'seed_' in fname:
            try:
                seed_str = fname.split('seed_')[1].split('_')[0]
                seed_idx = int(seed_str)
            except (IndexError, ValueError):
                continue
        else:
            continue  # Skip files without seed label

        try:
            df = pd.read_csv(f, header=None)
            vals = df.iloc[:, 1].astype(str).tolist()
        except Exception:
            continue

        # Convert to SMILES (no deduplication here, keep all)
        smi_list = []
        for s in vals:
            smi = _to_smiles(s, is_selfies=is_selfies)
            if smi is not None:
                smi_list.append(smi)

        if seed_idx not in seed_mols:
            seed_mols[seed_idx] = []
        seed_mols[seed_idx].extend(smi_list)

    # Apply limit per seed per experiment if specified
    if limit:
        for seed in seed_mols:
            if len(seed_mols[seed]) > limit:
                seed_mols[seed] = seed_mols[seed][:limit]

    return seed_mols


def compute_physchem_descriptors(smiles_list: List[str]) -> np.ndarray:
    """Compute a small set of physchem descriptors per SMILES.
    Returns array shape (N, D).
    Descriptors:
      - MolWt
      - TPSA
      - NumHBA
      - NumHBD
      - MolLogP (Crippen)
      - RingCount
      - FractionCSP3
      - NumRotatableBonds
      - HeavyAtomCount
      - NumAromaticRings
    """
    feats = []
    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                feats.append([0]*10)
                continue
            vals = [
                Descriptors.MolWt(mol),
                rdMolDescriptors.CalcTPSA(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                Crippen.MolLogP(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                Descriptors.HeavyAtomCount(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
            ]
            feats.append(vals)
        except Exception:
            feats.append([0]*10)
    return np.asarray(feats, dtype=np.float32)


def compute_morgan_fingerprints(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints using modern RDKit Generator API.
    Returns array shape (N, n_bits) with dtype float32.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            fp = gen.GetFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=np.int8)
            # Convert ExplicitBitVect to numpy
            DataStructs.ConvertToNumpyArray(fp, arr)
            out[i, :] = arr.astype(np.float32)
        except Exception:
            pass
    return out


def pca_then_embed(X: np.ndarray, method: str = 'umap', n_pca: int = 50, random_state: int = 0) -> np.ndarray:
    """Reduce high-dimensional features via PCA then UMAP/TSNE to 2D.
    Returns 2D coords (N, 2).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_samples = X.shape[0]
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # If all rows identical or too few samples, return zeros (or slight jitter)
    # if n_samples < 3 or np.allclose(X, X[0]):
    #     Z = np.zeros((n_samples, 2), dtype=np.float32)
    #     if n_samples > 1:
    #         Z += 1e-3 * np.random.RandomState(random_state).randn(n_samples, 2).astype(np.float32)
    #     return Z

    # PCA to reduce dimensionality safely
    # try:
    #     max_comps = max(2, min(n_pca, n_samples - 1, X.shape[1]))
    #     pca = PCA(n_components=max_comps, random_state=random_state)
    #     Xp = pca.fit_transform(X)
    # except Exception:
        # Fallback: use the original features
    Xp = X

    # UMAP if available, else TSNE with safe perplexity
    if method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            Z = reducer.fit_transform(Xp)
            return Z.astype(np.float32)
        except Exception:
            pass  # fall through to TSNE

    # TSNE requires perplexity < n_samples
    perplexity = max(2, min(30, n_samples - 1))
    try:
        tsne = TSNE(n_components=2, random_state=random_state, init='pca', learning_rate='auto', perplexity=perplexity)
        Z = tsne.fit_transform(Xp)
        return Z.astype(np.float32)
    except Exception:
        # Final fallback: take first two PCA components or pad with zeros
        if Xp.shape[1] >= 2:
            return Xp[:, :2].astype(np.float32)
        Z = np.zeros((n_samples, 2), dtype=np.float32)
        if Xp.shape[1] == 1:
            Z[:, 0] = Xp[:, 0].astype(np.float32)
        return Z


def combined_features(smiles_list: List[str], use_concat: bool = True) -> np.ndarray:
    """Compute fingerprints and physchem features; either concatenate or return fingerprints if no concat.
    """
    fp = compute_morgan_fingerprints(smiles_list)
    desc = compute_physchem_descriptors(smiles_list)
    if use_concat:
        return np.concatenate([fp, desc], axis=1)
    return fp


def collect_per_seed(repo: str, experiments: List[str], per_experiment_limit: Optional[int] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
    """Load and featurize molecules per seed across experiments.
    Returns {seed: (X, y, label_map)} where X is features, y is labels, label_map {experiment: label_id}
    """
    all_seeds = set()
    seed_data: Dict[int, Dict[str, List[str]]] = {}
    for exp in experiments:
        seed_mols = load_experiment_molecules_per_seed(repo, exp, limit=per_experiment_limit)
        for seed, mols in seed_mols.items():
            if seed not in seed_data:
                seed_data[seed] = {}
            seed_data[seed][exp] = mols
            all_seeds.add(seed)

    result: Dict[int, Tuple[np.ndarray, np.ndarray, Dict[str, int], List[str]]] = {}
    for seed in sorted(all_seeds):
        X_all = []
        y_all = []
        mol_list = []
        label_map: Dict[str, int] = {}
        label = 0
        for exp in experiments:
            mols = seed_data.get(seed, {}).get(exp, [])
            print(f"Seed {seed}, exp {exp}, len(mols) = {len(mols)}")
            if len(mols) > 0:
                print(f"  First 3 mols: {mols[:3]}")
            if len(mols) == 0:
                continue
            Fe = combined_features(mols, use_concat=False)
            X_all.append(Fe)
            y_all.append(np.full((Fe.shape[0],), label, dtype=np.int32))
            mol_list.extend(mols)
            label_map[exp] = label
            label += 1
        if len(X_all) == 0:
            continue
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        result[seed] = (X, y, label_map, mol_list)
    return result
