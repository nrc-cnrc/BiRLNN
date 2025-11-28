import re
from target_strings import smiles_list
# Base list of SMILES (edit this list to control what molecules to generate seeds from)
base_smiles = [smiles_list[i] for i in smiles_list.keys()]

seed_names = list(smiles_list.keys())

# Convert SMILES -> SELFIES (requires the 'selfies' package). Raise helpful error if missing.
try:
        import selfies as sf
except Exception as e:
        raise ImportError('The "selfies" package is required to convert SMILES to SELFIES. Install it with "pip install selfies"') from e

selfies_list = [sf.encoder(s) for s in base_smiles]


def make_forward(smiles: str) -> str:
        return 'G' + smiles


def make_forward_selfies(selfies: str) -> str:
        return '[G]' + selfies


def make_backward(smiles: str) -> str:
        return smiles + 'G'


def make_backward_selfies(selfies: str) -> str:
        return selfies + '[G]'


def make_bimodal(smiles: str) -> str:
        mid = len(smiles) // 2
        return smiles[:mid] + 'G' + smiles[mid:]


def make_bimodal_selfies(selfies: str) -> str:
    # split tokens of the form [..]
    tokens = re.findall(r'\[.*?\]', selfies)
    if not tokens:
        # Fallback: insert at character midpoint
        mid = len(selfies) // 2
        return selfies[:mid] + '[G]' + selfies[mid:]
    mid_idx = len(tokens) // 2
    tokens.insert(mid_idx, '[G]')
    return ''.join(tokens)


def seeds_for_model(model_name: str):
    # Return a list of seed strings appropriate for the given model name
    if 'ForwardRNN' in model_name:
        if 'SELFIES' in model_name:
            return [make_forward_selfies(s) for s in selfies_list]
        else:
            return [make_forward(s) for s in base_smiles]
    if 'BackwardRNN' in model_name:
        if 'SELFIES' in model_name:
            return [make_backward_selfies(s) for s in selfies_list]
        else:
            return [make_backward(s) for s in base_smiles]
    if 'BIMODAL' in model_name:
        if 'SELFIES' in model_name:
            return [make_bimodal_selfies(s) for s in selfies_list]
        else:
            return [make_bimodal(s) for s in base_smiles]
    if 'FBRNN' in model_name:
        if 'SELFIES' in model_name:
            return [make_bimodal_selfies(s) for s in selfies_list]
        else:
            return [make_bimodal(s) for s in base_smiles]

def get_seed_names():
    return seed_names