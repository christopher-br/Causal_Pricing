# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

################
# Load modules #
################

from tqdm import tqdm
import numpy as np
import logging
import warnings
import gc
import os
import sys
import itertools

DIR = ".../causal_pricing"
os.chdir(DIR)
sys.path.append(DIR)

from src.data.datagen import Data_object
from src.methods.mlp import MLP
from src.methods.drnet import DRNet
from src.methods.logreg import LogReg
from src.methods.randomf import RandomF
from src.methods.baseline import Baseline
from src.methods.gps import GPS
from src.methods.vcnet import VCNet

from src.utils.tune import tuner
from src.utils.setup import load_config, check_create_csv, get_rows, add_dict, add_row

############
# Settings #
############

RES_FILE = "results.csv"
TRACKER = "tracker.csv"
DATASET = "data/loan_data"

# Hyperparameters
HYPERPARAMS = load_config("config/methods/config.YAML")

# Number of iterations
n_iter = 10

# Save config file
CONFIG = load_config("config/data/config.YAML")["parameters"]

# Save para combinations
COMBINATIONS = list(itertools.product(*CONFIG.values()))


# Set terminal verbosity
terminal_verbose = False
if terminal_verbose == False:
        # configure logging at the root level of Lightning
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # configure logging on module level, redirect to file
        logger = logging.getLogger("pytorch_lightning.core")
        logger.addHandler(logging.FileHandler("core.log"))

        # Filter warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*overflow encountered.*")
        warnings.filterwarnings("ignore", ".*MPS available.*")
        warnings.filterwarnings("ignore", ".*on tensors of dimension other than 2 to reverse their shape is deprecated.*")
        warnings.filterwarnings("ignore", ".*lbfgs failed to converge.+")

# Disable automatic garbage collection
gc.disable()

##################
# Run experiment #
##################

# Create tracker
check_create_csv(TRACKER, CONFIG.keys())

for combination in tqdm(COMBINATIONS, desc="Iterate over combinations"):
    completed = get_rows(TRACKER)
    
    if (combination in completed):
        continue
    else:
        # Add combination to the tracker
        add_row(TRACKER, combination)
        # Save settings as a dictionary
        data_settings = dict(zip(CONFIG.keys(), combination))
        
    # Ini results
    results = {}
    results.update(data_settings)
    
    ################
    # Load dataset #
    ################
    
    # Save params
    dataset_params = {
            'confounding_bias': data_settings["bias"],
            'test_fraction': 0.2,
            'val_fraction': 0.1,
            'noise_std': 0.1,
            'dataset': DATASET,
            'gt': data_settings["gt"],
            'seed': data_settings["iteration"]
    }
    
    # Ini data
    obj = Data_object(dataset_params)
    dataset_train = obj.dataset_train
    dataset_val = obj.dataset_val
    dataset_test = obj.dataset_test
    
    # Save vars
    num_features = dataset_train['x'].shape[1]
    
    ############
    # Baseline #
    ############
    
    name = "baseline"
    
    # Params for tuning
    params = {
            'dummy': [1]
    }
    
    # Tune
    model = tuner(Baseline,
                    dataset_train,
                    dataset_val,
                    params,
                    'Baseline')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #######################
    # Logistic Regression #
    #######################
    
    name = "LogReg"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    
    # Tune
    model = tuner(LogReg,
                    dataset_train,
                    dataset_val,
                    params,
                    'Logistic Regression')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #################
    # Random Forest #
    #################
    
    name = "RandomF"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    
    # Tune
    model = tuner(RandomF,
                    dataset_train,
                    dataset_val,
                    params,
                    'Random Forest')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #######
    # MLP #
    #######
    
    name = "MLP"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    params.update({"inputSize": [num_features]})
    
    # Tune
    model = tuner(MLP,
                  dataset_train,
                  dataset_val,
                  params,
                  'MLP')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #######
    # GPS #
    #######
    
    name = "GPS"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    
    # Tune
    model = tuner(GPS,
                    dataset_train,
                    dataset_val,
                    params,
                    'GPS')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #########
    # DRNet #
    #########
    
    name = "DRNet"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    params.update({"inputSize": [num_features]})
    
    # Tune
    model = tuner(DRNet,
                    dataset_train,
                    dataset_val,
                    params,
                    'DRNet')
    
    # Evaluate
    mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #########
    # VCNet #
    #########
    
    name = "VCNet"
    
    # Params for tuning
    params = HYPERPARAMS[name]
    params.update({"inputSize": [num_features]})
    
    # Tune
    model = tuner(VCNet,
                    dataset_train,
                    dataset_val,
                    params,
                    'VCNet')
    
    # Bug fix if no convergence
    if model == None:
            mise, mise_r, pe, mr, bs = 99, 99, 99, 99, 99
    else:
            # Evaluate
            mise, mise_r, pe, mr, bs = model.computeMetrics(dataset_test)
    
    # Delete model
    del model
    
    # Log res
    results.update(
        {
            "MISE " + name: mise,
            "MISE R " + name: mise_r,
            "PE " + name: pe,
            "MR " + name: mr,
            "BS " + name: bs,
        }
    )
    
    # Run gc
    gc.collect()
    
    #################
    # End iteration #
    #################
    
    # Save results
    add_dict(RES_FILE, results)
    
    # Run gc
    gc.collect()