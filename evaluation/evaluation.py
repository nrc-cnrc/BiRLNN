"""
Implementation of the model evaluation

"""

import pandas as pd
import numpy as np
import configparser
import matplotlib
import sys
import os

# Make model/ importable
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, os.path.join(BASE_PATH, 'model'))
from helper import clean_molecule, check_valid

matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os


class Evaluator:
    def __init__(self, experiment_name, base_path=None):
        # Read parameter used during training
        self._config = configparser.ConfigParser()
        self._base_path = base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._config.read(os.path.join(self._base_path, 'experiments', experiment_name + '.ini'))

        self._model_type = self._config['MODEL']['model']
        self._experiment_name = experiment_name

        self._file_name = self._config['DATA']['data']
        self._encoding_size = int(self._config['DATA']['encoding_size'])
        self._molecular_size = int(self._config['DATA']['molecular_size'])

        self._epochs = int(self._config['TRAINING']['epochs'])
        self._n_folds = int(self._config['TRAINING']['n_folds'])
        self._learning_rate = float(self._config['TRAINING']['learning_rate'])
        self._batch_size = int(self._config['TRAINING']['batch_size'])

        self._samples = int(self._config['EVALUATION']['samples'])
        self._T = float(self._config['EVALUATION']['temp'])
        self._starting_token = self._config['EVALUATION']['starting_token']

        csv_path = os.path.join(self._base_path, 'data', self._file_name + '.csv')
        tar_path = os.path.join(self._base_path, 'data', self._file_name + '.tar.xz')
        if os.path.isfile(csv_path):
            self._data = pd.read_csv(csv_path, header=None).values[:, 0]
        elif os.path.isfile(tar_path):
            # Skip first line since empty and last line since nan
            self._data = pd.read_csv(tar_path, compression='xz', header=None).values[1:-1, 0]
        # Clean data from start, end and padding token
        self._fmt = 'SELFIES' if self._file_name.upper().startswith('SELFIES') else 'SMILES'
        for i, mol_dat in enumerate(self._data):
            self._data[i] = clean_molecule(mol_dat, self._model_type, fmt=self._fmt)

    def eval_training_validation(self, stor_dir='.'):
        '''Plot training and validation loss within one figure'''
        stat = np.zeros((self._n_folds, self._epochs))
        val = np.zeros((self._n_folds, self._epochs))
        plt.figure()
        plt.set_cmap('tab10')
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        for i in range(self._n_folds):
            tmp_path = os.path.join(stor_dir, self._experiment_name, 'statistic', f'stat_fold_{i+1}.csv')
            tmp = pd.read_csv(tmp_path, header=None).values[:, 1:]
            stat[i, :] = np.mean(tmp, axis=-1)
            plt.plot(np.arange(self._epochs) + 1, stat[i, :], '-', label=f'Fold {i+1} Training', color=f'C{i}')

        for i in range(self._n_folds):
            val_path = os.path.join(stor_dir, self._experiment_name, 'validation', f'val_fold_{i+1}.csv')
            val[i, :] = pd.read_csv(val_path, header=None).values[:, 1]
            plt.plot(np.arange(self._epochs) + 1, val[i, :], ':', label=f'Fold {i+1} Validation', color=f'C{i}')

        plt.legend()
        plt.title('Statistic Training')
        plt.ylabel('Loss Per Token')
        plt.xlabel('Epoch')
        out_path = os.path.join(stor_dir, self._experiment_name, 'statistic', 'all_statistic.png')
        plt.savefig(out_path)
        plt.close()

    def eval_training(self, stor_dir='.'):
        '''Plot training loss'''
        stat = np.zeros((self._n_folds, self._epochs))
        plt.figure(0)
        for i in range(self._n_folds):
            tmp_path = os.path.join(stor_dir, self._experiment_name, 'statistic', f'stat_fold_{i+1}.csv')
            tmp = pd.read_csv(tmp_path, header=None).values[:, 1:]
            stat[i, :] = np.mean(tmp, axis=-1)
            plt.plot(np.arange(1, self._epochs + 1), stat[i, :], label=f'Fold {i+1}')
        plt.legend()
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        out_path = os.path.join(stor_dir, self._experiment_name, 'statistic', 'statistic.png')
        plt.savefig(out_path)
        plt.close()

    def eval_validation(self, stor_dir='.'):
        '''Plot validation loss'''
        val = np.zeros((self._n_folds, self._epochs))
        plt.figure(1)
        for i in range(self._n_folds):
            val_path = os.path.join(stor_dir, self._experiment_name, 'validation', f'val_fold_{i+1}.csv')
            val[i, :] = pd.read_csv(val_path, header=None).values[:, 1]
            plt.plot(np.arange(1, self._epochs + 1), val[i, :], label=f'Fold {i+1}')
        plt.legend()
        plt.title('Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        out_path = os.path.join(stor_dir, self._experiment_name, 'validation', 'validation.png')
        plt.savefig(out_path)
        plt.close()

    def check_with_training_data(self, mol):
        '''Remove molecules that are within the training set and return remaining molecules'''
        to_delete = []
        for i, m in enumerate(mol):
            if m in self._data:
                to_delete.append(i)
        mol = np.delete(mol, to_delete)
        return mol

    def eval_molecule(self, stor_dir='.'):
        '''Plot percentage of novel, valid and unique SMILES'''
        valid = np.zeros((self._n_folds, self._epochs))
        unique = np.zeros((self._n_folds, self._epochs))
        novel = np.zeros((self._n_folds, self._epochs))

        for i in range(self._n_folds):
            for j in range(self._epochs):
                mol_path = os.path.join(stor_dir, self._experiment_name, 'molecules', f'molecule_fold_{i+1}_epochs_{j}.csv')
                mol = pd.read_csv(mol_path, header=None).values[:, 1].astype(str)

                # Remove padding
                # for k, m in enumerate(mol):
                    # molecules from CSV may be SELFIES; clean accordingly
                    # mol[k] = clean_molecule(m, self._model_type, fmt=self._fmt)
                # Compute unique molecules
                unique[i, j] = len(set(mol)) / self._samples

                # Remove duplicates
                mol = np.array(list(set(mol)))

                # Check validity and remove non-valid molecules
                to_delete = []
                for k, m in enumerate(mol):
                    if not check_valid(m, fmt=self._fmt):
                        to_delete.append(k)
                valid_mol = np.delete(mol, to_delete)
                valid[i, j] = len(valid_mol) / self._samples

                # Compute molecules unequal to training data
                if valid_mol.size != 0:
                    new_m = self.check_with_training_data(list(valid_mol))
                    novel[i, j] = len(new_m) / self._samples

        # Get percentage and statistics
        unique *= 100
        novel *= 100
        valid *= 100

        # Get mean values
        mean_unique = np.mean(unique, axis=0)
        mean_valid = np.mean(valid, axis=0)
        mean_novel = np.mean(novel, axis=0)

        # Get standard deviation
        std_unique = np.std(unique, axis=0)
        std_valid = np.std(valid, axis=0)
        std_novel = np.std(novel, axis=0)

        print(mean_unique)
        print(mean_valid)
        print(mean_novel)

        # PLot
        plt.figure(1)
        plt.errorbar(np.arange(1, self._epochs + 1), mean_unique, yerr=std_unique, capsize=3, label='unique')
        plt.errorbar(np.arange(1, self._epochs + 1), mean_valid, yerr=std_valid, capsize=3, label='valid & unique')
        plt.errorbar(np.arange(1, self._epochs + 1), mean_novel, yerr=std_novel, capsize=3, label='novel, valid & unique', linestyle=':')
        plt.yticks(np.arange(0, 110, step=10))
        plt.legend()
        plt.ylim(0, 105)
        plt.title(f'{self._fmt} T=' + str(self._T))
        plt.ylabel('% SMILES')
        plt.xlabel('Epoch')
        out_path = os.path.join(stor_dir, self._experiment_name, 'molecules', 'novel_valid_unique_molecules.pdf')
        plt.savefig(out_path)

        # Store data
        data = np.vstack((mean_unique, std_unique, mean_valid, std_valid, mean_novel, std_novel))
        csv_out = os.path.join(stor_dir, self._experiment_name, 'molecules', self._experiment_name + '_data.csv')
        pd.DataFrame(data).to_csv(csv_out)

        # plt.show()
