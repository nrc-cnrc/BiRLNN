"""
Implementation of different function simplifying SMILES and SELFIES handling
"""
import sys
import numpy as np
from rdkit import Chem
import os
import re
import selfies as sf
from rdkit.Chem import QED, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def get_qed(molecule):
    return QED.qed(molecule)

def get_sascore(molecule):
    return sascorer.calculateScore(molecule)

def clean_molecule(m, model_type, fmt='auto'):
    ''' Depending on the model different remains from generation should be removed
    :param m:   molecule with padding
    :param model_type:  Type of the model
    :return:    cleaned molecule
    '''
    # Ensure we work with a native Python string
    m = str(m)

    # Choose token forms depending on representation
    if model_type == 'FBRNN' or model_type == 'BIMODAL':
        m = remove_right_left_padding(m, start='G', end='E', fmt=fmt)
    elif model_type == 'ForwardRNN':
        # ForwardRNN sequences use right padding tokens (E) after the molecule
        m = remove_right_padding(m, end='E', fmt=fmt)
    elif model_type == 'BackwardRNN':
        # BackwardRNN sequences have the start token at the right and left-side padding (A)
        # so remove left padding instead of right padding
        m = remove_left_padding(m, pad='A', fmt=fmt)
    elif model_type == 'NADE':
        # NADE uses 'A' as padding token in the codebase
        m = remove_token([m], t='A', fmt=fmt)[0]
    else:
        print("CANNOT FIND MODEL")

    # Remove start token occurrences
    m = remove_token([m], 'G', fmt=fmt)
    m = remove_token(m, 'A', fmt=fmt)
    return m[0]


def remove_right_left_padding(mol, start='G', end='E', fmt='SMILES'):
    '''Remove right and left padding from start to end token
    :param mol:     SMILES or SELFIES string
    :param start:   token where to start
    :param end:     token where to finish
    :param fmt:     'SMILES' or 'SELFIES'
    :return:        new SMILES/SELFIES where padding is removed
    '''
    # SMILES behaviour: operate on raw string indices
    if fmt == 'SMILES':
        # Find start and end index
        mid_ind = mol.find(start)
        if mid_ind == -1:
            return ''
        end_ind = mol.find(end, mid_ind)
        if end_ind == -1:
            return ''
        # find last occurrence of end before mid_ind
        # fallback to start of string if not found
        rev_index = mol[::-1].find(end, len(mol) - mid_ind - 1)
        if rev_index == -1:
            start_ind = -1
        else:
            start_ind = len(mol) - rev_index - 1
        return mol[start_ind + 1:end_ind]

    # SELFIES behaviour: operate on bracket tokens
    elif fmt == 'SELFIES':
        toks = re.findall(r"\[.*?\]", mol)
        if len(toks) == 0:
            return ''
        # Ensure start/end are in bracket form
        s_tok = start if start.startswith('[') else f'[{start}]'
        e_tok = end if end.startswith('[') else f'[{end}]'

        # Find middle token occurrence
        try:
            mid_idx = toks.index(s_tok)
        except ValueError:
            # if no start token, return empty
            return ''

        # Find nearest end token to the left of mid_idx
        left_end = -1
        for i in range(mid_idx - 1, -1, -1):
            if toks[i] == e_tok:
                left_end = i
                break

        # Find nearest end token to the right of mid_idx
        right_end = -1
        for i in range(mid_idx + 1, len(toks)):
            if toks[i] == e_tok:
                right_end = i
                break

        # Determine slice
        start_idx = left_end + 1
        end_idx = right_end if right_end != -1 else len(toks)
        result_tokens = toks[start_idx:end_idx]
        return ''.join(result_tokens)


def remove_right_padding(mol, end='E', fmt='SMILES'):
    '''Remove right and left padding from start to end token
    :param mol:     SMILES string
    :param end:     token where to finish
    :return:        new SMILES where padding is removed'''
    if fmt == 'SMILES':
        end_ind = mol.find(end)
        if end_ind == -1:
            return mol
        return mol[:end_ind]

    elif fmt == 'SELFIES':
        # SELFIES: operate on bracket tokens
        toks = re.findall(r"\[.*?\]", mol)
        if len(toks) == 0:
            return ''
        e_tok = end if end.startswith('[') else f'[{end}]'
        # find first occurrence of end token
        for i, t in enumerate(toks):
            if t == e_tok:
                return ''.join(toks[:i])
        return ''.join(toks)


def remove_left_padding(mol, pad='A', fmt='SMILES'):
    '''Remove left padding tokens from SMILES or SELFIES.
    For SMILES this removes leading characters equal to `pad` (e.g. 'A').
    For SELFIES this removes leading bracketed tokens equal to `[pad]` (e.g. '[A]').
    :param mol: SMILES or SELFIES string
    :param pad: padding token (without brackets for SELFIES)
    :param fmt: 'SMILES' or 'SELFIES'
    :return: new SMILES/SELFIES string with left padding removed
    '''
    if fmt == 'SMILES':
        if mol is None:
            return ''
        # strip leading pad characters
        i = 0
        L = len(mol)
        while i < L and mol[i] == pad:
            i += 1
        result = mol[i:]
        # if the next token is an ending token 'E', remove it as well
        if result.startswith('E'):
            return result[1:]
        return result

    elif fmt == 'SELFIES':
        if mol is None:
            return ''
        toks = re.findall(r"\[.*?\]", mol)
        if len(toks) == 0:
            return ''
        pad_br = pad if pad.startswith('[') else f'[{pad}]'
        # find first token that is not the pad
        start_idx = 0
        while start_idx < len(toks) and toks[start_idx] == pad_br:
            start_idx += 1
        result_tokens = toks[start_idx:]
        # if the first remaining token is an ending token '[E]', remove it
        if len(result_tokens) > 0 and result_tokens[0] == '[E]':
            result_tokens = result_tokens[1:]
        return ''.join(result_tokens)
    
def check_valid(mol, fmt='auto'):
    '''Check if SMILES is valid
    :param mol:     SMILES string
    :return:        True / False
    '''
    # Empty molecule not accepted
    if mol == '':
        return False

    # If SELFIES, convert to SMILES first
    is_selfies = False
    if fmt == 'SELFIES':
        is_selfies = True
    elif fmt == 'SMILES':
        is_selfies = False
    else:
        raise ValueError("fmt must be 'SMILES' or 'SELFIES'")
    try:
        if is_selfies:
            smiles = sf.decoder(mol)
        else:
            smiles = mol
    except Exception:
        return False

    if smiles == '' or smiles is None:
        return False

    # Check valid with RDKit
    # MolFromSmiles returns None if molecule not valid
    rd_mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if rd_mol is None:
        return False
    return True


def remove_token(mol, t='G', fmt='SMILES'):
    '''Remove specific token from SMILES or SELFIES
    :param mol: list/array of strings
    :param t:   token to be removed (for SELFIES can be '[G]' or 'G')
    :param fmt: 'SMILES' or 'SELFIES'
    :return:    numpy array of strings with token removed
    '''
    out = []
    if fmt == 'SMILES':
        for d in mol:
            out.append(d.replace(t, ''))
        return out

    elif fmt == 'SELFIES':

        # SELFIES: ensure token is bracketed
        t_br = t if t.startswith('[') else f'[{t}]'
        for d in mol:
            if d is None:
                out.append('')
            else:
                out.append(str(d).replace(t_br, ''))
    return out


def check_model(model_type, model_name, stor_dir, fold, epoch):
    '''Perform fine-tuning and store statistic,
    :param stor_dir:    directory of stored data
    :param fold:    Fold to check
    :param epoch:   Epoch to check
    :return exists_model:   True if model exists otherwise False
    '''

    if model_type == 'NADE':
        exists_model = os.path.isfile(os.path.join(stor_dir, model_name, 'models', 'model_fold_' + str(fold) + '_epochs_' + str(epoch) + 'backdir.dat')) and \
                os.path.isfile(os.path.join(stor_dir, model_name, 'models', 'model_fold_' + str(fold) + '_epochs_' + str(epoch) + 'fordir.dat'))
    else:
        exists_model = os.path.isfile(os.path.join(stor_dir, model_name, 'models', 'model_fold_' + str(fold) + '_epochs_' + str(epoch) + '.dat'))

    return exists_model


def check_molecules(model_name, stor_dir, fold, epoch):
    '''Perform fine-tuning and store statistic,
    :param stor_dir:    directory of stored data
    :param fold:    Fold to check
    :param epoch:   Epoch to check
    :return :   True if molecules exist otherwise False
    '''

    return os.path.isfile(os.path.join(stor_dir, model_name, 'molecules', 'molecule_fold_' + str(fold) + '_epochs_' + str(epoch) + '.csv'))

if __name__ == '__main__':
    # simple test
    smi = 'GCCCTTTTECCO[F]EAAAA'
    cleaned = clean_molecule(smi, 'ForwardRNN', fmt='SMILES')
    print(f'Original: {smi}')
    print(f'Cleaned:  {cleaned}')