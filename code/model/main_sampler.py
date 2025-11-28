from sample import Sampler

for model in ['ForwardRNN_SELFIES_1024',
            'BackwardRNN_SELFIES_1024',
            'BIMODAL_SELFIES_fixed_1024',
            # 'FBRNN_SELFIES_fixed_1024',
            # 'BIMODAL_SELFIES_random_1024',
            # 'FBRNN_SELFIES_random_1024',
            'ForwardRNN_1024',
            'BackwardRNN_1024',
            'BIMODAL_fixed_1024', 
            # 'FBRNN_fixed_1024', 
            # 'BIMODAL_random_1024',
            # 'FBRNN_random_1024'
            ]:
# for model in ['ForwardRNN_1024']:
    s = Sampler(model)
    s.sample(fold=[1,2,3,4,5],N=200)
