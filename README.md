![repo version](https://img.shields.io/badge/Version-v.1.0-green)
![python version](https://img.shields.io/badge/python-%3E%3D3.11.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-%3E%3D2.8.0-orange)
![license](https://img.shields.io/badge/License-MIT-yellow.svg)

# BiRLNN: Bidirectional Reinforcement-Learning Neural Network for Constrained Molecular Design

This is the supporting code for: Junan L., Jiří H., Anguang H., Hang H., Hsu Kiang O., Mohammad Sajjad G., "BiRLNN: Bidirectional Reinforcement-Learning Neural Network for Constrained Molecular Design". Currently under review and available [here](https://www.researchsquare.com/article/rs-7540516/v1).

You can use this repository for both unconstrained and constrained generation of SMILES and SELFIES with unidirectional and bidirectional recurrent neural networks (RNNs). In addition to the methods' code, several pre-trained models for each approach are included.

The following methods are implemented:
* **Bidirectional Molecule Design by Alternate Learning** (BIMODAL), see [Grisoni *et al.* 2020](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00943).
* **Synchronous Forward Backward RNN** (FB-RNN), based on [Mou *et al.* 2016](https://arxiv.org/pdf/1512.06612.pdf).
* **Forward RNN**, *i.e.*, unidirectional RNN predicting in forward direction. 
* **Backward RNN**, *i.e.*, unidirectional RNN predicting in backward direction.


## Table of Contents
1. [Prerequisites](#Prerequisites)
2. [Using the Code](#Using_the_code)
  1. [Sampling from a pre-trained model](#Sample)
  2. [Training a model on your data](#Training) 
  3. [RL fine-tuning and evaluation](#Finetuning) 
  4. [Environment tips](#EnvTips)

3. [Authors](#Authors)
4. [License](#License)

## Prerequisites<a name="Prerequisites"></a>

This repository can be cloned with the following command:

```
git clone https://github.com/nrc-cnrc/BiRLNN
```

To install the necessary packages to run the code, we recommend using [conda](https://www.anaconda.com/download/). 
Once conda is installed, you can install the virtual environment:

```
cd path/to/repository/
conda env create -f birlnn.yml
```

To activate the dedicated environment:
```
conda activate birlnn
```

Your code should now be ready to use!

# Using the code <a name="Using_the_code"></a>
## Sampling from a pre-trained model <a name="Sample"></a>

In this repository, we provide you with 22 pre-trained models you can use for sampling (stored in [evaluation/](evaluation/)).
These models were trained on a set of 271,913 bioactive molecules from ChEMBL22 (K<sub>d/I</sub>/IC<sub>50</sub>/EC<sub>50</sub> <1μM), for 10 epochs.    

To sample SMILES, you can create a new file in [model/](model/) and use the *Sampler class*. 
For example, to sample from the pre-trained BIMODAL model with 512 units:

```
from sample import Sampler
experiment_name = 'BIMODAL_fixed_512'
s = Sampler(experiment_name)
s.sample(N=100, stor_dir='../evaluation', T=0.7, fold=[1], epoch=[9], valid=True, novel=True, unique=True, write_csv=True)
```

Parameters:
* *experiment_name* (str): name of the experiment with pre-trained model you want to sample from (you can find pre-trained models in [evaluation/](evaluation/))
* *stor_dir* (str): directory where the models are stored. The sampled SMILES will also be saved there (if write_csv=True)
* *N* (int): number of SMILES to sample
* *T* (float): sampling temperature
* *fold* (list of int): number of folds to use for sampling
* *epoch* (list of int): epoch(s) to use for sampling
* *valid* (bool): if set to *True*, only generate valid SMILES are accepted (increases the sampling time)
* *novel* (bool): if set to *True*, only generate novel SMILES (increases the sampling time)
* *unique* (bool): if set to *True*, only generate unique SMILES are provided (increases the sampling time)
* *write_csv* (bool): if set to *True*, the .csv file of the generated smiles will be exported in the specified directory.

*Notes*: 
- For the provided pre-trained models, only *fold=[1]* and *epoch=[9]* are provided.
- The list of available models and their description are provided in [evaluation/model_names.md](evaluation/model_names.md)

## Training a New Model
Alternatively, if you want to pre-train a model on your own data, you will need to execute three steps: (i) data processing (ii) training and (iii) evaluation.
Please be aware that you will need the access to a GPU to pre-train your own model as this is a computationally intensive step.

### Preprocessing
Data can be processed by using [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py):
```
from main_preprocessor import preprocess_data
preprocess_data(filename_in='../data/chembl_smiles', model_type='BIMODAL', starting_point='fixed', augmentation=1)
```
Parameters:
* *filename_in* (str): name of the file containing the SMILES strings (.csv or .tar.xz)
* *model_type* (str): name of the chosen generative method
* *starting_point* (str): starting point type ('fixed' or 'random')
* *augmentation*(int): augmentation folds [Default = 1]

*Notes*:
* In [preprocessing/main_preprocessor.py](preprocessing/main_preprocessor.py) you will find info regarding advanced options for pre-processing (e.g., stereochemistry, canonicalization, etc.)
* Please note that the pre-treated data will have to be stored in [data/](data/).

### Training

Training requires a parameter file (.ini) with a given set of parameters. You can find examples for all models in [experiments/](experiments/), and further details about the parameters below:


|Section		|Parameter     	| Description			|Comments|
| --- | --- | --- | --- |	
|Model		|model         	| Type				| ForwardRNN, BackwardRNN, FBRNN, BIMODAL  |
| 		|hidden_units	| Number of hidden units	|	Suggested value: 256 for ForwardRNN, BackwardRNN, FBRNN;  128 for BIMODAL|
|		|generation	| Defined through preprocessing 			| fixed, random |
|Data		|data		| Name of data file		| Has to be located in data/ |
| 		|encoding_size  | Number of different SMILES tokens		| 55 for SMILES, 75 for SELFIES |
|		|molecular_size	| Length of string with padding	| See preprocessing |
|Training	|epochs		| Number of epochs		|  Suggested value: 10 |
|		|learning_rate	| Learning rate			|  Suggested value: 0.001|
|		|n_folds	| Folds in cross-validation	| See below: More than 1 for cross_validation, 1 to use only one fold of the data for validation |
|		|batch_size	| Batch size			|  Suggested value: 128  |
|Evaluation	| samples	| Number of generated SMILES after each epoch |  |
|		| temp		| Sampling temperature		| Suggested value: 0.7 |
|		| starting_token	| Starting token for sampling	| G for SMILES models, [G] for SELFIES models	|


Options for training:

- Cross-validation: 
```
from trainer import Trainer

t = Trainer(experiment_name = 'BIMODAL_fixed_512')
t.cross_validation(stor_dir = '../evaluation/', restart = False)
```

- Single run: 1/*n_folds* of data used for validation
```
from trainer import Trainer

t = Trainer(experiment_name = 'BIMODAL_fixed_512')
t.single_run(stor_dir = '../evaluation/', restart = False)
```

Parameters:   
* *experiment_name* :  Name of parameter file (.ini)
* *stor_dir*: Directory where outputs can be found
* *restart*: If true, automatic restart from saved models (e.g. to be used if your training was interrupted before completion)

### Evaluation

You can do the evaluation of the outputs of your experiment with the [../evaluation/main_evaluator.py](../evaluation/main_evaluator.py) with the following possibilities:   

```
from evaluation import Evaluator

stor_dir = '../evaluation/'
e = Evaluator(experiment_name = 'BIMODAL_fixed_512')
# Plot training and validation loss within one figure
e.eval_training_validation(stor_dir=stor_dir)
# Plot percentage of novel, valid and unique SMILES
e.eval_molecule(stor_dir=stor_dir)
```

Parameters:
* *experiment_name*:  Name parameter file (.ini)
* *stor_dir*: Directory where outputs can be found

Note:
- the losses plot can be found, in that case, in '{experiment_name}/statistic/all_statistic.png'
- the novel, valid and unique SMILES plot can be found, in that case, in '../evaluation/{experiment_name}/molecules/novel_valid_unique_molecules.png'    

## Fine-tuning vis Reinforcement Learning <a name="Finetuning"></a>

This repo now includes a simple, model-agnostic REINFORCE-based RL fine-tuner to optimize molecular properties directly from the pre-trained generators. The entry point is `model/main_fine_tuner.py`.

The PPO fine-tuner (a clipped-objective variant that re-evaluates the current policy on the sampled sequences) is under development.

Reward formulation (consistent across both):

- Combined score = w_qed · QED − w_sas · (SAS − 1)/9
- Invalid molecules receive a penalty of −1.0

Quick start examples:

```bash
# REINFORCE for 200 episodes with checkpoints and per-episode generation
python model/main_fine_tuner.py \
  --algo reinforce --steps 200 --batch 40 --temp 0.7 --fold 2 --epoch 9 \
  --reward-weights 1,1 \
  --ckpt-every 20 \
  --gen-episodes 0,100,200 \
  --gen-per-episode 1000

# PPO (experimental) for 100 episodes
python model/main_fine_tuner.py \
  --algo ppo --steps 100 --batch 40 --temp 0.7 --fold 2 --epoch 9 \
  --reward-weights 1,1
```

### Output folders and files

All RL outputs are organized under `evaluation/rl/<algo>/` in a per-run directory keyed by reward weights and optional run name, for example:

- `evaluation/rl/reinforce/weights_1_1/`
  - `reinforce_trajectory.pdf`: per-episode combined score curve with ±1σ band (REINFORCE)
  - `checkpoints/<EXPERIMENT>/`: model checkpoints saved at step 0 (initial) and every K episodes
  - `generated_molecules_epoch{EP}.csv`: per-episode CSVs of sampled molecules with columns:
    - `SELFIES` (if applicable), `SMILES`, `QED_Score`, `N_heavy`, `SA_Score`
  - `distribution.pdf` and `distribution.svg`: legacy-style QED vs SAS distribution plots (see below)
  - `top_molecules/epoch_{ep}_rank_{k}.svg`: top molecules per episode (highest QED and lowest SAS), annotated

- `evaluation/rl/ppo/weights_1_1/`
  - `trajectory.png`: PPO combined score curve

Notes:
- Checkpoints are saved using the underlying model’s native `.dat` format (same as evaluation models).
- When using PPO, checkpoints are placed under `evaluation/rl/checkpoints/<EXPERIMENT>/`.

### How to read the plots

1) RL trajectory (`reinforce_trajectory.pdf`)

- X-axis: episode index; Y-axis: combined reward (QED − SAS/9 with weights).
- The shaded band shows the within-batch standard deviation; a rising mean with narrowing band indicates more stable improvements.

2) QED vs SAS distribution (`distribution.pdf`/`.svg`)

- Main panel: scatter of sampled molecules per episode overlayed.
- Marginal KDEs (top/right): normalized density per episode to compare shifts across runs.
- Dashed lines: per-episode means for QED (top) and SAS (right).
- Desired movement: up (higher QED) and left (lower SAS).

3) Top-molecule SVGs (`top_molecules/epoch_{ep}_rank_{k}.svg`)

- Per-episode thumbnails of the best candidates.
- Sorted by high QED and low SAS; file name encodes episode and rank.
- SVGs include annotations for QED and SA to quickly skim improvements.

### Tips

- Temperature: 0.6–0.8 is a good starting range; lower tends to exploit, higher explores.
- Batch size: 32–64 works well on a single GPU; larger batches stabilize gradients.
- Reward weights: `(1,0)` pushes QED only; `(1,1)` balances QED up and SAS down.
- If you modify tokenization or start tokens, ensure `helper.clean_molecule` matches your model type.


## Environment tips: PyTorch vision ops and C++ ABI <a name="EnvTips"></a>

If you run into errors like “operator torchvision::nms does not exist” or “version `CXXABI_1.3.15` not found” when using AiZynthFinder, SciPy, or other compiled extensions, use the following checklist.

- Keep torch and torchvision matched from the same source and CUDA version.
  - Example (CUDA 12.4, torch 2.5.1): install torchvision 0.20.1+cu124 from the PyTorch wheel index.
  - Avoid mixing conda and pip for these two; install both from either the PyTorch wheels or the pytorch/conda channel consistently.
- Prefer the conda C++ runtime (libstdc++) ahead of system libraries to satisfy modern CXXABI symbols required by SciPy and others.
  - Prepend your environment’s lib directory to LD_LIBRARY_PATH when launching analysis scripts.

Optional commands (Linux, bash):

```bash
# Make sure torchvision matches torch and CUDA (example: torch 2.5.1+cu124)
conda remove -y torchvision torchvision-extra-decoders
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torchvision==0.20.1

# Ensure conda’s libstdc++ is preferred when running tools that import SciPy/AiZynthFinder
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

Notes:
- torch 2.5.1 pairs with torchvision 0.20.x. If you upgrade torch, also upgrade torchvision to the matching series.
- If you prefer conda-only installation, install torch/torchvision from the pytorch channel with a matching CUDA runtime, or use CPU-only builds consistently.

## Retrosynthesis with AiZynthFinder: getting useful solvability estimates

We include a helper in `model/retrosynthesis_aizynth.py` that evaluates retrosynthetic solvability per episode for generated molecules. To get meaningful results:

- Provide a valid AiZynthFinder config YAML with expansion policies, ringbreaker, filter policy, and a realistic stock of purchasable building blocks. Example keys in `config.yml`:
  - expansion.uspto, expansion.ringbreaker: ONNX models + template CSVs
  - filter.uspto: ONNX filter model
  - stock.zinc: a purchasable stock HDF5
- The analyzer now does the following automatically for each run:
  - Selects all expansion/filter policies and the provided stock
  - Tunes search to return the first found route (saves time on positives)
  - Moderately increases the iteration budget when the default is very low

You can control runtime vs. depth using the per-molecule time limit:

```python
from model.retrosynthesis_aizynth import analyze_and_plot_retro_for_run

analyze_and_plot_retro_for_run(
    run_dir="evaluation/rl/reinforce/weights_1_1",
    episodes=[0, 100, 200],
    config_path="config.yml",
    max_molecules=None,          # or an integer for a quick sample
    per_mol_timeout=5.0,         # seconds per target; set higher for deeper search
)
```

Practical guidance:
- If you see zero solved routes, first verify that the expansion/filter models and the stock paths in your config actually exist and load (we ship example paths in `config.yml`).
- For quick, broad estimates, start with `per_mol_timeout=1-5` seconds and iterate. For higher recall, increase to 10–30 seconds and consider raising `iteration_limit` (see `retrosynthesis_aizynth._configure_finder`).
- Route solvability is sensitive to the stock. A richer stock (e.g., ZINC, Enamine) and good ringbreaker templates generally increase solve rates.

## Authors<a name="Authors"></a>

* Junan Lin (https://github.com/JunanLin)

## License<a name="License"></a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This code is licensed under the MIT License — see the LICENSE file for details.

© 2025 The authors. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
