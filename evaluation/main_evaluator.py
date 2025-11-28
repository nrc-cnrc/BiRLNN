from evaluation import Evaluator
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
stor_dir = os.path.join(base_path, 'evaluation')

for name in [
    'BIMODAL_SELFIES_fixed_1024',
    'BIMODAL_SELFIES_random_1024',
    'FBRNN_SELFIES_fixed_1024',
    'FBRNN_SELFIES_random_1024',
    'ForwardRNN_SELFIES_1024',
    'BackwardRNN_SELFIES_1024',
    'BIMODAL_fixed_1024',
    'BIMODAL_random_1024',
    'FBRNN_fixed_1024',
    'FBRNN_random_1024',
    'ForwardRNN_1024',
    'BackwardRNN_1024'
    ]:
    e = Evaluator(experiment_name=name, base_path=base_path)
    # evaluation of training and validation losses
    e.eval_training_validation(stor_dir=stor_dir)
    # evaluation of sampled molecules
    e.eval_molecule(stor_dir=stor_dir)