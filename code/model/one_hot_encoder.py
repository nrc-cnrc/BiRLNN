"""
One-hot encoders for SMILES and SELFIES.

Provides a small, compatible API used by the training and sampling code:
 - SMILESEncoder.encode(data: np.ndarray[str]) -> np.ndarray (N, L, T)
 - SMILESEncoder.decode(one_hot: np.ndarray) -> np.ndarray[str]
 - SELFIESEncoder.encode / decode with analogous shapes

Both encoders expose `pad_index` (int or None) which is the category index
for the padding token ('A' for SMILES, '[A]' for SELFIES) when present.
"""
import os
import sys
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class SMILESEncoder:
    def __init__(self):
        # Allowed tokens (deterministic ordering)
        self._tokens = np.array([
            '#', '=', '\\', '/', '%', '@', '+', '-', '.',
            '(', ')', '[', ']',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'A', 'B', 'E', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V',
            'Z',
            'a', 'b', 'c', 'd', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't'
        ], dtype=object)

        self._encoder = OneHotEncoder(categories=[self._tokens], dtype=np.uint8)
        # index of padding token 'A' in self._tokens
        try:
            self.pad_index = int(np.where(self._tokens == 'A')[0][0])
        except Exception:
            self.pad_index = None

    def encode_from_file(self, name: str = 'data') -> np.ndarray:
        # Read CSV or compressed tar.xz as in the original project
        if os.path.isfile(name + '.csv'):
            data = pd.read_csv(name + '.csv', header=None).values
        elif os.path.isfile(name + '.tar.xz'):
            data = pd.read_csv(name + '.tar.xz', compression='xz', header=None).values[1:-1]
        else:
            print('CAN NOT READ DATA')
            sys.exit(1)

        shape = data.shape
        data = data.reshape(-1)
        data = np.squeeze(data)
        return self.encode(data).reshape((shape[0], shape[1], -1, len(self._tokens)))

    def encode(self, data: Sequence[str]) -> np.ndarray:
        # Split SMILES strings into characters (assumes fixed-length sequences per file)
        char_data = self.smiles_to_char(np.asarray(data, dtype=str))
        shape = char_data.shape
        flat = char_data.reshape((-1, 1))

        transformed = self._encoder.fit_transform(flat)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        transformed = transformed.reshape((shape[0], shape[1], -1))
        return transformed

    def decode(self, one_hot: np.ndarray) -> np.ndarray:
        shape = one_hot.shape[0]
        one_hot = one_hot.reshape((-1, len(self._tokens)))
        # Some sampling routines may produce rows that are entirely zero
        # (no token selected). sklearn's OneHotEncoder.inverse_transform
        # will raise in that case when handle_unknown='error'. Replace
        # any all-zero rows with the padding token if available, otherwise
        # raise a clearer error.
        if one_hot.size == 0:
            data = np.empty((0, 0), dtype=object)
        else:
            row_sums = one_hot.sum(axis=1)
            zero_mask = (row_sums == 0)
            if zero_mask.any():
                if self.pad_index is not None:
                    one_hot[zero_mask, :] = 0
                    one_hot[zero_mask, self.pad_index] = 1
                else:
                    raise ValueError("Encountered all-zero token rows during decode and no pad_index is set.")

            data = self._encoder.inverse_transform(one_hot)
        data = data.reshape((shape, -1))
        smiles = self.char_to_smiles(data)
        return np.array(smiles)

    @staticmethod
    def smiles_to_char(data: Sequence[str]) -> np.ndarray:
        # Split SMILES into characters and pad to same length using 'A' as pad
        lists = [list(s) for s in data]
        max_len = max(len(x) for x in lists)
        padded = []
        for lst in lists:
            padded.append(np.array(lst, dtype=object))
        return np.stack(padded, axis=0)

    @staticmethod
    def char_to_smiles(char_data: np.ndarray) -> np.ndarray:
        out = []
        for i in range(char_data.shape[0]):
            out.append(''.join(char_data[i, :]))
        return np.array(out)


class SELFIESEncoder:
    def __init__(self):
        # Use canonical, full token list (deterministic ordering)
        self._tokens = np.array([
            '[#11C]', '[#Branch1]', '[#Branch2]', '[#C-1]', '[#C]', '[#N+1]', '[#N]', '[11CH3]', '[123I]', '[125I]',
            '[13CH1]', '[13C]', '[17F]', '[18F]', '[19F]', '[2H]', '[3H]', '[76Br]', '[=Branch1]', '[=Branch2]',
            '[=C]', '[=N+1]', '[=N-1]', '[=NH1+1]', '[=NH2+1]', '[=N]', '[=O+1]', '[=O]', '[=P]', '[=Ring1]', '[=Ring2]',
            '[=S+1]', '[=S]', '[=Se]', '[A]', '[As]', '[B-1]', '[BH2-1]', '[BH3-1]', '[B]', '[Br]', '[Branch1]',
            '[Branch2]', '[C-1]', '[C]', '[Cl]', '[E]', '[F]', '[G]', '[I]', '[N+1]', '[N-1]', '[NH1+1]', '[NH1-1]',
            '[NH1]', '[NH2+1]', '[NH3+1]', '[N]', '[O-1]', '[OH0]', '[OH1+1]', '[O]', '[P+1]', '[PH1]', '[P]',
            '[Ring1]', '[Ring2]', '[S+1]', '[S-1]', '[S]', '[Se+1]', '[SeH1]', '[Se]', '[Si]', '[Te]'
        ], dtype=object)
        # Initialize OneHotEncoder with fixed categories
        self._encoder = OneHotEncoder(categories=[self._tokens], dtype=np.uint8)
        # padding index for '[A]' if present
        try:
            self.pad_index = int(np.where(self._tokens == '[A]')[0][0])
        except Exception:
            self.pad_index = None
        # pattern to split SELFIES into bracketed tokens
        self._pattern = re.compile(r"\[.*?\]")

    def encode_from_file(self, name: str = 'data') -> np.ndarray:
        if os.path.isfile(name + '.csv'):
            data = pd.read_csv(name + '.csv', header=None).values
        elif os.path.isfile(name + '.tar.xz'):
            data = pd.read_csv(name + '.tar.xz', compression='xz', header=None).values[1:-1]
        else:
            print('CAN NOT READ DATA')
            sys.exit(1)

        shape = data.shape
        data = data.reshape(-1)
        data = np.squeeze(data)
        return self.encode(data).reshape((shape[0], shape[1], -1, len(self._tokens)))

    def encode(self, data: Sequence[str]) -> np.ndarray:
        token_data = self.selfies_to_tokens(np.asarray(data, dtype=str))
        shape = token_data.shape
        flat = token_data.reshape((-1, 1))

        # Use the fixed encoder (categories were set in __init__)
        transformed = self._encoder.fit_transform(flat)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        transformed = transformed.reshape((shape[0], shape[1], -1))
        return transformed

    def decode(self, one_hot: np.ndarray) -> np.ndarray:
        if self._tokens is None or self._encoder is None:
            raise ValueError('Encoder has not been fitted yet.')
        shape = one_hot.shape[0]
        one_hot = one_hot.reshape((-1, len(self._tokens)))
        # Handle all-zero rows produced during sampling by mapping them to the
        # pad token ('[A]') if available. This avoids sklearn raising a
        # confusing ValueError inside inverse_transform.
        if one_hot.size == 0:
            data = np.empty((0, 0), dtype=object)
        else:
            row_sums = one_hot.sum(axis=1)
            zero_mask = (row_sums == 0)
            if zero_mask.any():
                if self.pad_index is not None:
                    one_hot[zero_mask, :] = 0
                    one_hot[zero_mask, self.pad_index] = 1
                else:
                    raise ValueError("Encountered all-zero token rows during decode and no pad_index is set.")

            data = self._encoder.inverse_transform(one_hot)
        data = data.reshape((shape, -1))
        selfies = self.tokens_to_selfies(data)
        return np.array(selfies)

    def selfies_to_tokens(self, data: Sequence[str]) -> np.ndarray:
        # Tokenize SELFIES and pad to equal length using '[A]' as pad.
        # Determine token lists per string (count of bracketed tokens) rather than string length.
        lists = []
        for s in data:
            # Ensure we operate on string values; handle NaN/None gracefully
            if s is None:
                toks = []
            else:
                toks = self._pattern.findall(str(s))
            lists.append(toks)

        # max_len is the maximum number of bracketed tokens across samples
        max_len = max((len(x) for x in lists), default=0)

        padded = []
        for toks in lists:
            if len(toks) < max_len:
                toks = toks + ['[A]'] * (max_len - len(toks))
            padded.append(np.array(toks, dtype=object))

        # If all sequences are empty, return an empty array with shape (n, 0)
        if max_len == 0:
            return np.zeros((len(lists), 0), dtype=object)

        return np.stack(padded, axis=0)

    @staticmethod
    def tokens_to_selfies(token_data: np.ndarray) -> np.ndarray:
        out = []
        for i in range(token_data.shape[0]):
            out.append(''.join(token_data[i, :]))
        return np.array(out)

