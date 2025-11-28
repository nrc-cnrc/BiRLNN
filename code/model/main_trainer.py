import sys, os
import time
repo='/home/jameshko/Documents/birlnn_latest/BIMODAL'
# Ensure model package imports find local modules
sys.path.insert(0, os.path.join(repo, 'model'))
from trainer import Trainer

for exp in [
    # 'FBRNN_SELFIES_fixed_1024', 
    # 'FBRNN_SELFIES_random_1024', 
    'FBRNN_fixed_1024', 
    # 'FBRNN_random_1024', 
    ]:
# for exp in ['BIMODAL_SELFIES_random_512_quick']:
# for exp in ['BackwardRNN_SELFIES_512_quick']:
# for exp in ['ForwardRNN_SELFIES_1024','BackwardRNN_SELFIES_1024']:
    print('\n==== Starting training for', exp, '====')
    start_time = time.time()
    t = Trainer(exp, base_path=repo)
    t.cross_validation()
    # t.single_run()
    end_time = time.time()
    print('==== Finished training for', exp, '====')
    print(f"Training time: {(end_time - start_time)/3600:.2f} hours")