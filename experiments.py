import pdb
import sys


def run_experiments(dataset, args):
    from LogicCBM.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y,
            train_X_to_Cy,
            train_X_to_bC_to_Ly,
        )
        
    experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'Multitask':
        train_X_to_Cy(*args)
    
    elif experiment == 'Logic':
        train_X_to_bC_to_Ly(*args)

def parse_arguments():
    print(sys.argv[2])
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    assert sys.argv[1].upper() in ['OAI', 'CUB', 'CIFAR100', 'AWA2', 'IMAGENET100', 'IMAGENET50'], 'Please specify OAI or CUB dataset'
    assert sys.argv[2] in ['Concept_XtoC', 'Independent_CtoY', 'Sequential_CtoY',
                           'Standard', 'StandardWithAuxC', 'Multitask', 'Joint', 'Logic', 'Probe',
                           'TTI', 'Robustness', 'HyperparameterSearch'], \
        'Please specify valid experiment. Current: %s' % sys.argv[2]
    
    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()
    
    from LogicCBM.train import parse_arguments
    args = parse_arguments(experiment=experiment)
    return dataset, args

if __name__ == '__main__':

    import torch
    import numpy as np

    dataset, args = parse_arguments()

    # Seeds
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)

    run_experiments(dataset, args)
