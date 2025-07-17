
import random
from typing import List, Tuple

import numpy as np

from xdream.experiment.utils.misc import ref_code_recovery
from xdream.experiment.utils.parsing import get_rnd
from xdream.experiment.utils.args import REFERENCES, ExperimentArgParams
from xdream.core.utils.io_ import load_pickle
from xdream.core.utils.misc import copy_exec
from xdream.core.utils.parameters import ArgParams
from xdream.core.subject import TorchNetworkSubject

def generate_log_numbers(N, M): return list(sorted(list(set([int(a) for a in np.logspace(0, np.log10(M), N)]))))


NAME   = f'13_07_optvar_moreelitism'  

ITER     = 500              # number of opt iterations                          
SAMPLE   =  10              # number of experiments ran per neuron




# -- OPTIM VARIANTS --
OPT_VARIANTS = ['genetic', 'hybrid', 'cmaes'] 
OPT_VARIANT_LAYER = 56                                      # which layer to record, 56 is the last (classfication layer)
OPT_VARIANT_NEURONS = list(random.sample(range(1000), 10))  # how many neurons to record, 10 random neurons from the last layer
NET             = 'resnet50'                                # network name

# -- HYBRID VARIANTS --

PARAMS_VARIANTS = [(
        "set1",  
        {
            "inertia_max": 0.99,
                "inertia_min": 0.95,
                "cognitive": 2.5,
                "social": 2.0,
                "v_clip": 0.15,
                "num_informants": 20,
                "mut_size": 0.4,
                "mut_rate": 0.4,
                "first_PSO_interval": 0.6,
                "second_PSO_interval": 0.85,
        }
    ),
    (
        "set2",
        {
            'inertia_max': 0.99,
                    'inertia_min': 0.95,
                    'cognitive': 2.5,
                    'social': 2,
                    'v_clip': 0.15,
                    'num_informants': 15,
                    "mut_size": 0.5,
                    "mut_rate": 0.5,
                    'first_PSO_interval': 0.6,
                    'second_PSO_interval': 0.9,   
        }
    ),
    (
        "set3",
        {
            "inertia_max": 0.99,
                    "inertia_min": 0.90,
                    "cognitive": 2,
                    "social": 2.0,
                    "v_clip": 0.15,
                    "num_informants": 20,
                    "mut_size": 0.1,
                    "mut_rate": 0.1,
                    "first_PSO_interval": 0.6,
                    "second_PSO_interval": 0.8,
        }
    ),
    (
    "set4",
    {
        "inertia_max": 0.99,
                    "inertia_min": 0.95,
                    "cognitive": 2.0,
                    "social": 2.0,
                    "v_clip": 0.15,
                    "num_informants": 15,
                    "mut_size": 0.6,
                    "mut_rate": 0.2,
                    "first_PSO_interval": 0.6,
                    "second_PSO_interval": 0.85,
        }
    ),
]




def get_args_optimizer_variants() -> Tuple[str, str, str]:

    args = [
        (f'{OPT_VARIANT_LAYER}=[{neuron}]', f'{variant}', str(random.randint(1000, 1000000)) )
        for neuron in OPT_VARIANT_NEURONS
        for variant in OPT_VARIANTS
        for _ in range(SAMPLE)
    ]
    
    rec_str        = '#'.join(a for a, _, _ in args)
    variant_str    = '#'.join(a for _, a, _ in args)
    rand_seed_str  = '#'.join(a for _, _, a in args)
    
    return rec_str, variant_str, rand_seed_str

def get_args_OptPararms_variants() -> Tuple[str, str, str]:
    combinations = []
    
    # Generate all combinations
    for param_set_name, param_dict in PARAMS_VARIANTS:
        for neuron in OPT_VARIANT_NEURONS:
            for _ in range(SAMPLE):
                combinations.append({
                    "recording_layer": f'{OPT_VARIANT_LAYER}=[{neuron}]',
                    "optimizer_type": f"hybrid:{param_set_name}", 
                    "random_seed": str(random.randint(1000, 1000000)),
                })
    
    # Create the arguments
    args = [
        (comb["recording_layer"], comb["optimizer_type"], comb["random_seed"])
        for comb in combinations
    ]
    
    rec_str = '#'.join(a for a, _, _ in args)
    optim_type_str = '#'.join(a for _, a, _ in args)
    rand_seed_str = '#'.join(a for _, _, a in args)
    
    return rec_str, optim_type_str, rand_seed_str






if __name__ == '__main__':

    print('Multiple run: ')
 
    print('[1] Optimizer variants')
    print('[2] OptParams variants')
    
    option = int(input('Choose option: '))
    
    match option:
        
        
            
        case 1:
            
            rec_layer_str, variant_str, rnd_seed_str = get_args_optimizer_variants()
            
            args = {
                str(ExperimentArgParams.RecordingLayers) : rec_layer_str,
                str(ExperimentArgParams.ScoringLayers  ) : f'{OPT_VARIANT_LAYER}=[]',
                str(ExperimentArgParams.OptimType      ) : variant_str,
                str(          ArgParams.RandomSeed     ) : rnd_seed_str,
                str(ExperimentArgParams.NetworkName    ) : NET
            }
            
            file = 'run_multiple_optimizer_variants.py'
            
            
        case 2: 
            
       

            rec_str, optim_type_str, rnd_seed_str = get_args_OptPararms_variants()
            args = {
                str(ExperimentArgParams.RecordingLayers): rec_str,
                str(ExperimentArgParams.OptimType): optim_type_str, 
                str(ArgParams.RandomSeed): rnd_seed_str,
                str(ExperimentArgParams.ScoringLayers): f"{OPT_VARIANT_LAYER}=[]",
                str(ExperimentArgParams.NetworkName): NET
            }
            file = 'run_multiple_optParams_variants.py'
            
        
        case _:
            
            print('Invalid option')
            
    args[str(ArgParams          .ExperimentName)] = NAME
    args[str(ArgParams          .NumIterations )] = str(ITER)
    args[str(ExperimentArgParams.Template      )] = 'T'
    

    copy_exec(file=file, args=args )