import os
import math
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

"""AiZynthFinder-based retrosynthesis analysis utilities (UNDER DEVELOPMENT)."""

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _try_get_steps_from_route(route) -> Optional[int]:
    """Best-effort extraction of the number of reaction steps from an AiZynthFinder route.

    Tries common attribute names and dict exports. Returns None if not available.
    """
    # Common attribute names
    for name in ("reaction_count", "reactions_count", "nb_steps", "n_steps", "number_of_steps"):
        if hasattr(route, name):
            try:
                val = getattr(route, name)
                return int(val()) if callable(val) else int(val)
            except Exception:
                pass
    # Reactions list length
    for name in ("reactions", "steps", "nodes"):
        if hasattr(route, name):
            try:
                val = getattr(route, name)
                if isinstance(val, (list, tuple)):
                    return int(len(val))
            except Exception:
                pass
    # Try dict export
    try:
        if hasattr(route, "to_dict"):
            d = route.to_dict()
            for key in ("nreactions", "n_reactions", "steps", "n_steps"):
                if key in d and d[key] is not None:
                    return int(d[key])
    except Exception:
        pass
    return None


def _configure_finder(
    finder,
    per_mol_timeout: Optional[float] = None,
) -> None:
    """Configure AiZynthFinder instance with sensible defaults for our batch use.

    - Select all expansion and filter policies defined in the config
    - Select all available stocks
    - Optionally set per-molecule time limit (seconds)
    - Prefer returning the first found route to save time on positives
    - Bump iteration limit moderately to improve coverage vs. default (100)
    """
    # Ensure models/stocks are active
    try:
        if hasattr(finder, "expansion_policy"):
            finder.expansion_policy.select_all()
        if hasattr(finder, "filter_policy"):
            finder.filter_policy.select_all()
        if hasattr(finder, "stock"):
            finder.stock.select_all()
    except Exception:
        # Non-fatal: if selection APIs differ, continue with config defaults
        pass

    # Tune search behavior if available
    cfg = getattr(finder, "config", None)
    search = getattr(cfg, "search", None) if cfg is not None else None
    if search is not None:
        # Stop at first route to keep runtime bounded on solvable targets
        try:
            if hasattr(search, "return_first"):
                setattr(search, "return_first", True)
        except Exception:
            pass
        # Give the search a bit more room than default (100)
        try:
            if hasattr(search, "iteration_limit"):
                # Only bump if still at a very low default
                if int(getattr(search, "iteration_limit", 100)) <= 100:
                    setattr(search, "iteration_limit", 1000)
        except Exception:
            pass
        # Respect caller-provided per-molecule timeout if set
        if per_mol_timeout is not None:
            try:
                if hasattr(search, "time_limit"):
                    setattr(search, "time_limit", float(per_mol_timeout))
            except Exception:
                pass


def run_aizynth_on_smiles(
    smiles_list: List[str],
    config_path: str,
    max_molecules: Optional[int] = None,
    per_mol_timeout: Optional[float] = None,
) -> List[Dict[str, Optional[float]]]:
    """Run AiZynthFinder on a list of SMILES strings and return route stats per molecule.

    Returns list of dicts with: {'SMILES', 'RouteFound' (0/1), 'MinSteps' (int or None)}
    """
    try:
        from aizynthfinder.aizynthfinder import AiZynthFinder
    except Exception as e:
        raise RuntimeError(
            "AiZynthFinder is not installed. Please install it and provide a valid config YAML."
        ) from e

    finder = AiZynthFinder(configfile=config_path)
    _configure_finder(finder, per_mol_timeout=per_mol_timeout)

    results = []
    n = len(smiles_list) if max_molecules is None else min(len(smiles_list), int(max_molecules))
    iterator = range(n)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="AiZynthFinder", total=n, leave=True)
    for i in iterator:
        smi = (smiles_list[i] or "").strip()
        if not smi:
            results.append({"SMILES": smi, "RouteFound": 0, "MinSteps": None})
            continue
        try:
            finder.target_smiles = smi
            start = time.time()
            finder.tree_search()
            # Note: time limit is handled by finder.config.search.time_limit when set

            routes_obj = getattr(finder, "routes", None)
            route_list = None
            if routes_obj is None:
                found = 0
                min_steps = None
            else:
                # Get list of route objects
                if hasattr(routes_obj, "routes"):
                    route_list = routes_obj.routes
                else:
                    try:
                        route_list = list(routes_obj)
                    except Exception:
                        route_list = None
                if route_list is None:
                    found = 0
                    min_steps = None
                else:
                    found = 1 if len(route_list) > 0 else 0
                    if found:
                        steps_vals = []
                        for r in route_list:
                            st = _try_get_steps_from_route(r)
                            if st is not None:
                                steps_vals.append(int(st))
                        min_steps = int(min(steps_vals)) if steps_vals else None
                    else:
                        min_steps = None
            results.append({"SMILES": smi, "RouteFound": found, "MinSteps": min_steps})
        except Exception:
            # Treat errors as unsolved
            results.append({"SMILES": smi, "RouteFound": 0, "MinSteps": None})
    return results


def analyze_and_plot_retro_for_run(
    run_dir: str,
    episodes: List[int],
    config_path: str,
    smiles_col: str = "SMILES",
    max_molecules: Optional[int] = None,
    per_mol_timeout: Optional[float] = None,
) -> None:
    """Analyze AiZynthFinder routes for generated CSVs in a run directory and save summary + violin plot.

    Expects per-episode CSVs named generated_molecules_epoch{ep}.csv with a SMILES column.
    Outputs under run_dir/retrosynthesis/:
      - episode_{ep}_routes.csv with RouteFound and MinSteps
      - summary.csv (episode, n_total, n_solved, frac_solved)
      - steps_violin.pdf (violin plot of MinSteps for solved molecules across episodes)
    """
    os.makedirs(os.path.join(run_dir, "retrosynthesis"), exist_ok=True)
    summary_rows = []
    steps_data = []  # list of (episode, steps) for solved molecules

    for ep in episodes:
        csv_path = os.path.join(run_dir, f"generated_molecules_epoch{ep}.csv")
        if not os.path.exists(csv_path):
            print(f"[retro] Missing CSV for episode {ep}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            # Try common alternatives
            for alt in ("SMILES", "smiles", "canonical_smiles"):
                if alt in df.columns:
                    smiles_col = alt
                    break
        smiles = df[smiles_col].fillna("").astype(str).tolist()
        res = run_aizynth_on_smiles(
            smiles,
            config_path=config_path,
            max_molecules=max_molecules,
            per_mol_timeout=per_mol_timeout,
        )
        out_df = pd.DataFrame(res)
        out_path = os.path.join(run_dir, "retrosynthesis", f"episode_{ep}_routes.csv")
        out_df.to_csv(out_path, index=False)
        n_total = len(out_df)
        n_solved = int(out_df["RouteFound"].sum()) if "RouteFound" in out_df.columns else 0
        frac = (n_solved / n_total) if n_total > 0 else 0.0
        summary_rows.append({"episode": ep, "n_total": n_total, "n_solved": n_solved, "frac_solved": frac})

        # Accumulate MinSteps for solved molecules
        solved_steps = out_df.loc[(out_df["RouteFound"] == 1) & out_df["MinSteps"].notna(), "MinSteps"].astype(int).tolist()
        for st in solved_steps:
            steps_data.append((ep, st))

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("episode").to_csv(
            os.path.join(run_dir, "retrosynthesis", "summary.csv"), index=False
        )

    # Violin plot of steps distribution for solved molecules across episodes
    if steps_data:
        # Use non-interactive backend to avoid display/lib issues
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
        except Exception as e:
            print('[retro] Matplotlib unavailable, skipping violin plot:', e)
            return
        # Organize by episode
        ep_to_steps: Dict[int, List[int]] = {}
        for ep, st in steps_data:
            ep_to_steps.setdefault(ep, []).append(int(st))
        ordered_eps = sorted(ep_to_steps.keys())
        data = [ep_to_steps[ep] for ep in ordered_eps]

        plt.figure(figsize=(6, 4))
        parts = plt.violinplot(data, showmeans=True, showextrema=True)
        plt.xticks(np.arange(1, len(ordered_eps) + 1), [str(e) for e in ordered_eps])
        plt.xlabel("Episode")
        plt.ylabel("Shortest route steps (solved)")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "retrosynthesis", "steps_violin.pdf"))
        plt.close()
