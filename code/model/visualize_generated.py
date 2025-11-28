"""
Visualization utilities for generated molecules.

Functions:
 - plot_qed_vs_sas: joint scatter + marginal KDEs across epochs
 - plot_normalized_distribution: KDEs of a quantity across epochs and optionally training data
 - plot_evolution_stats: plot mean +/- std of a property across epochs
 - save_top_molecule_images: save SVG images of top-n molecules by sorting criteria

"""

import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw




def _read_generated_csv(model_dir: str, epoch: int) -> Optional[pd.DataFrame]:
    csv_file = os.path.join(model_dir, f"generated_molecules_epoch{epoch}.csv")
    if not os.path.exists(csv_file):
        print(f"[visualize] File not found: {csv_file}")
        return None
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"[visualize] Failed to read {csv_file}: {e}")
        return None


def plot_qed_vs_sas(model_dir: str, episodes: List[int], qed_threshold: float = 1e-8, sa_threshold: float = 9.99, output_name: str = "distribution"):
    """Create a joint scatter plot (QED vs SA) with marginal KDEs for specified epochs.

    Saves `distribution.pdf` and `distribution.svg` in `model_dir`.
    """
    sns.set_theme(style='white')
    # Use seaborn context to scale fonts and also set explicit rc keys so
    # axes/tick/legend sizes are increased reliably for saved figures.
    sns.set_context('notebook', font_scale=1.4)
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
    })
    grid = sns.JointGrid(height=6, ratio=5)

    colors = sns.color_palette(n_colors=max(3, len(episodes)))

    any_plotted = False
    for i, episode in enumerate(episodes):
        df = _read_generated_csv(model_dir, episode)
        if df is None or df.empty:
            continue
        # apply thresholds
        filtered = df[(df.get("QED_Score") >= qed_threshold) & (df.get("SA_Score") <= sa_threshold)]
        if filtered.empty:
            continue
        any_plotted = True
        color = colors[i % len(colors)]
        # main scatter
        grid.ax_joint.scatter(filtered["QED_Score"], filtered["SA_Score"], color=color, alpha=0.6, marker='.', label=f"Episode {episode}")
        # marginal KDEs (normalized)
        sns.kdeplot(filtered["QED_Score"], ax=grid.ax_marg_x, color=color, fill=True, alpha=0.3)
        sns.kdeplot(y=filtered["SA_Score"], ax=grid.ax_marg_y, color=color, fill=True, alpha=0.3)
        # mean lines on marginals
        try:
            qed_mean = float(pd.to_numeric(filtered["QED_Score"], errors='coerce').dropna().mean())
            sa_mean = float(pd.to_numeric(filtered["SA_Score"], errors='coerce').dropna().mean())
            grid.ax_marg_x.axvline(qed_mean, color=color, linestyle='--', alpha=0.8)
            grid.ax_marg_y.axhline(sa_mean, color=color, linestyle='--', alpha=0.8)
        except Exception:
            pass

    if not any_plotted:
        print("[visualize] No data plotted: check model_dir and epoch CSV files.")
        return

    grid.ax_joint.set_xlabel("QED Score")
    grid.ax_joint.set_ylabel("SA Score")
    grid.ax_joint.legend()
    plt.tight_layout()

    out_pdf = os.path.join(model_dir, f"{output_name}.pdf")
    out_svg = os.path.join(model_dir, f"{output_name}.svg")
    plt.savefig(out_pdf)
    plt.savefig(out_svg)
    print(f"[visualize] Saved plots: {out_pdf}, {out_svg}")
    plt.close()


def plot_normalized_distribution(model_dir: str, epochs: List[int], quantity: str, training_data_file: Optional[str] = None, output_name: Optional[str] = None):
    """Plot KDEs of `quantity` for generated epochs and optionally training data.

    Saves `{quantity}_distribution.pdf|svg` under model_dir unless `output_name` provided.
    """
    sns.set(style='whitegrid')
    sns.set_context('notebook', font_scale=1.4)
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
    })
    plt.figure(figsize=(8, 5))

    if training_data_file and os.path.exists(training_data_file):
        try:
            train_df = pd.read_csv(training_data_file)
            if quantity in train_df:
                sns.kdeplot(train_df[quantity], label='training', color='black', fill=True, alpha=0.25)
        except Exception as e:
            print(f"[visualize] Failed to read training data {training_data_file}: {e}")

    colors = sns.color_palette(n_colors=max(3, len(epochs)))
    plotted = False
    for i, epoch in enumerate(epochs):
        df = _read_generated_csv(model_dir, epoch)
        if df is None or df.empty or quantity not in df:
            continue
        sns.kdeplot(df[quantity].dropna(), label=f"epoch {epoch}", color=colors[i % len(colors)], fill=True, alpha=0.25)
        plotted = True

    if not plotted and (training_data_file is None or not os.path.exists(training_data_file)):
        print("[visualize] Nothing to plot: check files and quantity name.")
        return

    plt.xlabel(quantity)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    out_name = output_name or f"{quantity}_distribution"
    out_pdf = os.path.join(model_dir, f"{out_name}.pdf")
    out_svg = os.path.join(model_dir, f"{out_name}.svg")
    plt.savefig(out_pdf)
    plt.savefig(out_svg)
    print(f"[visualize] Saved distribution plots: {out_pdf}, {out_svg}")
    plt.close()


def plot_evolution_stats(model_dir: str, epochs: List[int], quantity: str = "QED_Score", output_name: Optional[str] = None):
    """Plot mean +/- std of `quantity` across epochs.

    Expects CSV files `generated_molecules_epoch{epoch}.csv` in `model_dir`.
    """
    means = []
    stds = []
    valid_epochs = []
    for epoch in epochs:
        df = _read_generated_csv(model_dir, epoch)
        if df is None or df.empty or quantity not in df:
            continue
        vals = pd.to_numeric(df[quantity], errors='coerce').dropna()
        if vals.empty:
            continue
        means.append(vals.mean())
        stds.append(vals.std())
        valid_epochs.append(epoch)

    if not valid_epochs:
        print("[visualize] No valid epochs found for evolution plotting.")
        return

    sns.set_context('notebook', font_scale=1.4)
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
    })
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(valid_epochs, means, yerr=stds, fmt='-o')
    plt.xlabel('Episode')
    plt.ylabel(quantity)
    plt.title(f'Mean +/- STD of {quantity} over epochs')
    plt.tight_layout()

    out_name = output_name or f"evolution_{quantity}"
    out_pdf = os.path.join(model_dir, f"{out_name}.pdf")
    out_svg = os.path.join(model_dir, f"{out_name}.svg")
    plt.savefig(out_pdf)
    plt.savefig(out_svg)
    print(f"[visualize] Saved evolution plots: {out_pdf}, {out_svg}")
    plt.close()


def save_top_molecule_images(model_dir: str, epochs: List[int], criteria: List[str], ascending: List[bool], n: int = 20):
    """Save SVG images of top `n` molecules per epoch sorted by `criteria`.

    criteria: list of column names to sort by (e.g., ['QED_Score', 'SA_Score'])
    ascending: list of booleans for sort order corresponding to criteria
    """
    for epoch in epochs:
        df = _read_generated_csv(model_dir, epoch)
        if df is None or df.empty:
            continue
        if 'SMILES' not in df.columns:
            print(f"[visualize] epoch {epoch} CSV lacks 'SMILES' column; skipping")
            continue
        # drop duplicates and NaNs
        df = df.dropna(subset=['SMILES']).drop_duplicates(subset=['SMILES'])
        # ensure criteria exist
        for c in criteria:
            if c not in df.columns:
                print(f"[visualize] Column {c} not found in epoch {epoch}; skipping")
                continue
        sorted_df = df.sort_values(by=criteria, ascending=ascending).head(n)

        out_dir = os.path.join(model_dir, f"epoch_{epoch}_top_molecules")
        os.makedirs(out_dir, exist_ok=True)

        for rank, row in enumerate(sorted_df.itertuples(), start=1):
            smiles = getattr(row, 'SMILES')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[visualize] Invalid SMILES at epoch {epoch} rank {rank}: {smiles}")
                continue
            try:
                # Match legacy style: 300x360, two lines with QED/SA values
                qed_score = getattr(row, 'QED_Score', None)
                sa_score = getattr(row, 'SA_Score', None)
                drawer = Draw.MolDraw2DSVG(300, 360)
                drawer.DrawMolecule(mol)
                drawer.FinishDrawing()
                svg = drawer.GetDrawingText()
                annotation = ""
                if qed_score is not None:
                    annotation += f"<text x='60' y='330' font-size='20' fill='black'>QED Score: {qed_score:.5f}</text>"
                if sa_score is not None:
                    annotation += f"<text x='60' y='350' font-size='20' fill='black'>SA Score: {sa_score:.5f}</text>"
                svg = svg.replace('</svg>', f"{annotation}</svg>")
                out_file = os.path.join(out_dir, f"epoch_{epoch}_rank_{rank}.svg")
                with open(out_file, 'w') as f:
                    f.write(svg)
                print(f"[visualize] Saved {out_file}")
            except Exception as e:
                print(f"[visualize] Failed to draw molecule {smiles}: {e}")

if __name__ == "__main__":
    plot_qed_vs_sas(
        model_dir="/home/jameshko/Documents/birlnn_latest/BIMODAL/evaluation/rl/FBRNN_SELFIES_fixed_1024_copy/reinforce/weights_1_0/",
        episodes=[0, 100, 200],
        qed_threshold=0.0,
        sa_threshold=10.0,
        output_name="qed_vs_sa_distribution"
    )
