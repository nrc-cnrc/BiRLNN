"""
Implementation of the sampler to generate SMILES from a trained model

"""

import pandas as pd
import numpy as np
import configparser
from fb_rnn import FBRNN
from forward_rnn import ForwardRNN
from backward_rnn import BackwardRNN
from one_hot_encoder import SMILESEncoder, SELFIESEncoder
from bimodal import BIMODAL
import os
from helper import clean_molecule, check_valid

# np.random.seed(1)


class Sampler():

    def __init__(self, experiment_name, base_path=None):
        # Read parameter used during training
        self._config = configparser.ConfigParser()
        # determine repository root (parent of model/)
        self._base_path = base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._config.read(os.path.join(self._base_path, 'experiments', experiment_name + '.ini'))

        # Track absolute paths for any CSVs written by sample(); persists across calls
        self._output_files = []  # type: list[str]

        self._model_type = self._config['MODEL']['model']
        self._experiment_name = experiment_name
        self._hidden_units = int(self._config['MODEL']['hidden_units'])

        self._file_name = self._config['DATA']['data']
        # choose encoder based on data filename prefix like Trainer does
        if self._file_name.upper().startswith('SELFIES'):
            self._encoder = SELFIESEncoder()
        else:
            self._encoder = SMILESEncoder()
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])

        self._epochs = int(self._config['TRAINING']['epochs'])
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        # Optional seed file path (list of initial strings), else single starting_token
        self._seed_file = None
        if 'starting_token_file' in self._config['EVALUATION']:
            raw = self._config['EVALUATION']['starting_token_file']
            self._seed_file = raw.strip() if raw is not None else None
            if self._seed_file == '':
                self._seed_file = None
        # Pre-encode single starting token if provided (may be multi-character)
        self._starting_token = None
        if 'starting_token' in self._config['EVALUATION']:
            st = self._config['EVALUATION']['starting_token']
            if st is not None and str(st).strip() != '':
                # Encoder returns shape (1, L, vocab)
                self._starting_token = self._encoder.encode([st])[0]

        pad_idx = getattr(self._encoder, 'pad_index', None)
        if self._model_type == 'FBRNN':
            self._model = FBRNN(self._molecular_size, self._encoding_size,
                                self._learning_rate, self._hidden_units, pad_index=pad_idx)
        elif self._model_type == 'ForwardRNN':
            self._model = ForwardRNN(self._molecular_size, self._encoding_size,
                                     self._learning_rate, self._hidden_units, pad_index=pad_idx)

        elif self._model_type == 'BIMODAL':
            self._model = BIMODAL(self._molecular_size, self._encoding_size,
                                  self._learning_rate, self._hidden_units, pad_index=pad_idx)
        elif self._model_type == 'BackwardRNN':
            self._model = BackwardRNN(self._molecular_size, self._encoding_size,
                                      self._learning_rate, self._hidden_units, pad_index=pad_idx)

        # Read data (use absolute paths resolved from base path)
        csv_path = os.path.join(self._base_path, 'data', self._file_name + '.csv')
        tar_path = os.path.join(self._base_path, 'data', self._file_name + '.tar.xz')
        if os.path.isfile(csv_path):
            self._data = pd.read_csv(csv_path, header=None).values[:, 0]
        elif os.path.isfile(tar_path):
            # Skip first line since empty and last line since nan
            self._data = pd.read_csv(tar_path, compression='xz', header=None).values[1:-1, 0]

        # Clean data from start, end and padding token
        # Determine data format
        fmt = 'SELFIES' if self._file_name.upper().startswith('SELFIES') else 'SMILES'
        for i, mol_dat in enumerate(self._data):
            self._data[i] = clean_molecule(mol_dat, self._model_type, fmt=fmt)

    def sample(self, N=100, stor_dir=None, T=0.7, fold=[1], epoch=[9], valid=True, novel=True, unique=True, write_csv=True, base_path=None,
               seeds_file=None, seeds=None, per_seed=False, seeds_label=None):

        '''Sample from a model where the number of novel valid unique molecules is fixed
        :param stor_dir:    directory where the generated SMILES are saved
        :param N:        number of samples
        :param T:        Temperature
        :param fold:     Folds to use for sampling
        :param epoch:    Epochs to use for sampling
        :param valid:    If True, only accept valid SMILES
        :param novel:    If True, only accept novel SMILES
        :param unique:   If True, only accept unique SMILES
        :param write_csv If True, the generated SMILES are written in stor_dir
        :return: res_molecules: list with all the generated SMILES
        '''

        # Resolve storage directory: prefer explicit stor_dir, else use base_path/evaluation
        base = base_path or self._base_path
        if stor_dir is None:
            stor_dir = os.path.join(base, 'evaluation')
        elif not os.path.isabs(stor_dir):
            stor_dir = os.path.join(base, stor_dir)
        fmt = 'SELFIES' if self._file_name.upper().startswith('SELFIES') else 'SMILES'
        res_molecules = []
        print('Sampling: started')
        for f in fold:
            for e in epoch:
                self._model.build(os.path.join(stor_dir, self._experiment_name, 'models',
                                               'model_fold_' + str(f) + '_epochs_' + str(e)))

                # Resolve seed strings:
                # Priority: seeds argument > seeds_file argument > config starting_token_file > single starting_token
                seed_strings = None
                # explicit seeds list of strings
                if seeds is not None:
                    seed_strings = list(seeds)
                else:
                    # path resolution helper
                    def _resolve_path(p):
                        if p is None:
                            return None
                        if os.path.isabs(p):
                            return p
                        return os.path.join(base, p)

                    file_path = seeds_file or self._seed_file
                    if file_path is not None:
                        file_path = _resolve_path(file_path)
                        if os.path.isfile(file_path):
                            with open(file_path, 'r') as fh:
                                # One seed string per line; strip empties
                                seed_strings = [ln.strip() for ln in fh if ln.strip() != '']
                        else:
                            raise FileNotFoundError(f"Seeds file not found: {file_path}")
                # If still None, fall back to single starting_token from config
                if seed_strings is None:
                    if self._starting_token is None:
                        raise ValueError("No starting token or seeds file provided in config or arguments.")
                    seed_strings = [self._config['EVALUATION']['starting_token']]

                # Encode all seeds (handles SMILES or SELFIES automatically via encoder)
                # encoder.encode expects list[str] -> (N, L, vocab). We'll keep individual (L, vocab) per seed.
                encoded_list = []
                # Encode in one batch to fit encoder categories
                enc_all = self._encoder.encode(seed_strings)
                for i in range(enc_all.shape[0]):
                    encoded_list.append(enc_all[i])

                # Determine how many samples to generate
                if per_seed:
                    target_total = N * len(encoded_list)
                else:
                    target_total = N

                new_molecules = []
                # Round-robin through seeds until reaching target_total
                idx = 0
                while len(new_molecules) < target_total:
                    seed_enc = encoded_list[idx % len(encoded_list)]
                    # Model.sample accepts (vocab,) or (L, vocab); we pass (L, vocab)
                    new_mol = self._encoder.decode(self._model.sample(seed_enc, T))

                    # Remove remains from generation
                    
                    new_mol = clean_molecule(new_mol[0], self._model_type, fmt=fmt)

                    # If not valid, get new molecule
                    if valid and not check_valid(new_mol, fmt=fmt):
                        continue

                    # If not unique, get new molecule
                    if unique and (new_mol in new_molecules):
                        continue

                    # If not novel, get molecule
                    if novel and (new_mol in self._data):
                        continue

                    # If all conditions checked, add new molecule
                    new_molecules.append(new_mol)

                # Prepare name for file
                name = fmt + '_molecules_fold_' + str(f) + '_epochs_' + str(e) + '_T_' + str(T)
                if per_seed:
                    name += f"_per_seed_{N}"
                else:
                    name += f"_N_{N}"
                name += '.csv'
                if unique:
                    name = 'unique_' + name
                if valid:
                    name = 'valid_' + name
                if novel:
                    name = 'novel_' + name
                if seeds_label is not None:
                    name = f"seed_{seeds_label}_" + name

                # Store final molecules
                if write_csv:
                    mol_dir = os.path.join(stor_dir, self._experiment_name, 'molecules')
                    if not os.path.exists(mol_dir):
                        os.makedirs(mol_dir)
                    mol = np.array(new_molecules).reshape(-1)
                    out_path = os.path.join(mol_dir, name)
                    pd.DataFrame(mol).to_csv(out_path, header=None)
                    # Record absolute path to written CSV
                    self._output_files.append(os.path.abspath(out_path))
        
            res_molecules.append(new_molecules)
        
        print('Sampling: done')
        return res_molecules

    def clear_output_files(self):
        """Clear the tracked list of generated CSV absolute file paths."""
        self._output_files = []

    @property
    def output_files(self):
        """List[str]: Absolute paths of CSV files written by calls to sample()."""
        return list(self._output_files)
        
