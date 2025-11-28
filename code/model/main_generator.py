import sys
import os
import time
import re

repo = '/home/jameshko/Documents/birlnn_latest/BIMODAL'
# Ensure model package imports find local modules
sys.path.insert(0, os.path.join(repo, 'model'))
from sample import Sampler
from seeds import seeds_for_model

for model in [ 
        'FBRNN_fixed_1024',
        'FBRNN_SELFIES_fixed_1024',
        'FBRNN_random_1024',
        'FBRNN_SELFIES_random_1024'
        ]:
        seeds = seeds_for_model(model)
        for i, seed in enumerate(seeds):
                s = Sampler(model, base_path=repo)
                time_start = time.time()
                s.sample(N=200, fold=[1, 2, 3, 4, 5], epoch=[9], seeds=[seed], unique=True, valid=True, novel=True, seeds_label=i)
                time_end = time.time()
                print(f"Sampling time for model {model} and seed index {i}: {(time_end - time_start)/60} minutes")
