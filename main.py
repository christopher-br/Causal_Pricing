# Pipeline by Christopher Bockel-Rickermann. Copyright (c) 2023

# For run in SageMaker:
# In shell run 'python /home/ec2-user/SageMaker/01_Experiment/main.py'
# In line 102 change dataset

# Key variations tbd in data_prepper.py, e.g.,
# - Shape of dr curve (l. 348) --> Primary
# - Shape of selection bias (l. 284) --> Secondary 

# Preliminaries in SageMaker:
import os
# os.chdir('/home/ec2-user/SageMaker/01_Experiment')

##################################################
# Main script
##################################################

# Save main run parameters
n_iter = 10
n_passes = 5000
biases = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
terminal_verbose = False
log_results_locally = False

# Load modules
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import logging
import warnings
import gc
import copy

# Load data class and DRNets
# For detailed intro to DRNets see original paper by Schwab (2020)
# https://doi.org/10.1609/aaai.v34i04.6014
from modules.adaDRNet import DRNet_Model
from modules.data_prepper import Data_object, get_dataset_splits

# Load benchmark models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load eval, log and print utils
from utils.print_utils import get_dr_SKL, get_true_dr, print_summary_plot, get_treatment_strengths
from utils.eval_utils import compute_metrics_SKL, compute_metrics_baseline
from utils.wandb_utils import getAPI

# Log into WandB
wandb.login(key=getAPI())
wandb.init(project='CausalPricing', entity='br1ckmann', reinit=False)

# Save experiments
experiments = ["DRNets", "MLP", "Linear regression",
               "Logistic regression", "Random forests", "Baseline"]
# Save metrics
metrics = ["MISE", "MISE Revenue", "PE", "MSFE", "MR"]

# Ini results dict
results = dict()
for metric in metrics:
    results[metric] = dict()
    for experiment in experiments:
        results[metric][experiment] = []

# Counter for logging
step_id = 0

# Set terminal verbosity and logging behaviour
if (terminal_verbose == False):
    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler("core.log"))
    
    # Filter warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*overflow encountered.*")
    warnings.filterwarnings("ignore", ".*MPS available.*")
    
# Disable garbage collection
gc.disable()

#########
# Iterate
#########

for bias in tqdm(biases, desc="Iterating over biases", ncols=125):
    for n in tqdm(range(n_iter), leave=False, desc="Iteration", ncols=125):
        # Log step details
        wandb.log({"Bias": bias,
                   "Iteration": n},
                  step=step_id)
        
        ##############
        # Load dataset
        ##############
                
        # Save dataset parameters
        dataset_params = dict()
        dataset_params['num_treatments'] = int(1)
        dataset_params['treatment_selection_bias'] = float(0)
        dataset_params['dosage_selection_bias'] = float(bias)
        dataset_params['test_fraction'] = float(0.20)
        dataset_params['val_fraction'] = float(0.10)
        # Added args
        dataset_params['dataset'] = str("datasets/to be specified")
        dataset_params['include_cat_vars'] = True
        dataset_params['binary_targets'] = True
        dataset_params['noise_std'] = float(0.1)
        dataset_params['seed'] = n
        
        # Initialize data
        data_class = Data_object(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        num_features = dataset_train['x'].shape[1]
        
        # Calculate average observation for plotting
        mean_observation = np.mean(dataset_test['x'], axis=0)
        
        ##########
        # Training
        ##########
        
        # MLP
        #*#*#*#*#*#*#*#*#*
        
        # Ini validation error
        val_error = np.inf
        
        # Ini pbar
        pbar = tqdm(desc="Train MLP", total=12, leave=False, ncols=125)
        
        # Tune
        for b_size in [64, 128, 256]:
            for n_heads in [1]: # For basic MLP
                for l_rate in [0.01, 0.1]:
                    for h_size in [64, 128]:
                        for n_steps in [n_passes]:
                            for n_layers in [2]:                                
                                # Save config file
                                config = dict()
                                config["modelName"] = "MLP"
                                config["learningRate"] = l_rate
                                config["batchSize"] = b_size
                                config["hiddenSize"] = h_size
                                config["numSteps"] = n_steps
                                config["numLayers"] = n_layers
                                config["numHeads"] = n_heads
                                config["inputSize"] = num_features
                                
                                # Set up model
                                temp = DRNet_Model(config)
                                # Train
                                temp.trainModel(dataset_train)
                                # Get validation error
                                temp_error = temp.validateModel(dataset_test)
                                # Save new lowest error and save model
                                if (temp_error < val_error):
                                    val_error = temp_error
                                    model = copy.deepcopy(temp)
                                        
                                # Update pbar
                                pbar.update(1)
        
        # Close pbar
        pbar.close()
                                    
        # Evaluate
        mise, mise_r, pe, msfe, mr = model.compute_metrics(dataset_test)
        
        # Get dr curve
        mlp_dr, mlp_drr = model.get_dr(mean_observation)
        
        # Del previous models
        del model
        # Delete temp
        del temp
        
        # Append to results array
        results["MISE"]["MLP"].append(mise)
        results["MISE Revenue"]["MLP"].append(mise_r)
        results["PE"]["MLP"].append(pe)
        results["MSFE"]["MLP"].append(msfe)
        results["MR"]["MLP"].append(mr)
        
        # Log results
        wandb.log({"MISE MLP": mise,
                   "MISE Revenue MLP": mise_r,
                   "PE MLP": pe,
                   "MSFE MLP": msfe,
                   "MR MLP": mr},
                  step=step_id)
        
        # DRNets
        #*#*#*#*#*#*#*#*#*
        
        # Ini validation error
        val_error = np.inf
        
        # Ini pbar
        pbar = tqdm(desc="Train DRNets", total=24, leave=False, ncols=125)
        
        # Tune
        for b_size in [64, 128, 256]:
            for n_heads in [5, 10]:
                for l_rate in [0.01, 0.1]:
                    for h_size in [64, 128]:
                        for n_steps in [n_passes]:
                            for n_layers in [2]:                                
                                # Save config file
                                config = dict()
                                config["modelName"] = "DRNets"
                                config["learningRate"] = l_rate
                                config["batchSize"] = b_size
                                config["hiddenSize"] = h_size
                                config["numSteps"] = n_steps
                                config["numLayers"] = n_layers
                                config["numHeads"] = n_heads
                                config["inputSize"] = num_features
                                
                                # Set up model
                                temp = DRNet_Model(config)
                                # Train
                                temp.trainModel(dataset_train)
                                # Get validation error
                                temp_error = temp.validateModel(dataset_test)
                                # Save new lowest error and save model
                                if (temp_error < val_error):
                                    val_error = temp_error
                                    model = copy.deepcopy(temp)
                                        
                                # Update pbar
                                pbar.update(1)
        
        # Close pbar
        pbar.close()
                                    
        # Evaluate
        mise, mise_r, pe, msfe, mr = model.compute_metrics(dataset_test)
        
        # Get dr curve
        drnets_dr, drnets_drr = model.get_dr(mean_observation)
        
        # Del previous models
        del model
        # Delete temp
        del temp
        
        # Append to results array
        results["MISE"]["DRNets"].append(mise)
        results["MISE Revenue"]["DRNets"].append(mise_r)
        results["PE"]["DRNets"].append(pe)
        results["MSFE"]["DRNets"].append(msfe)
        results["MR"]["DRNets"].append(mr)
        
        # Log results
        wandb.log({"MISE DRNets": mise,
                   "MISE Revenue DRNets": mise_r,
                   "PE DRNets": pe,
                   "MSFE DRNets": msfe,
                   "MR DRNets": mr},
                  step=step_id)
        
        #*#*#*#*#*#*#*#*#*
        # SKLearn methods
        #*#*#*#*#*#*#*#*#*
        
        # Transform train data
        Train_X = np.column_stack((dataset_train["x"],
                                   dataset_train["d"].reshape((-1, 1))))
        Train_Y = dataset_train["y"]
        # ...test data
        Test_X = np.column_stack((dataset_test["x"],
                                  dataset_test["d"].reshape((-1, 1))))
        Test_Y = dataset_test["y"]
        
        # Lin regression
        #*#*#*#*#*#*#*#*#*
        
        # Ini model
        linreg_model = LinearRegression()

        # Train model
        linreg_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                         dataset_test=dataset_test,
                                                         model=linreg_model)

        # Get dr
        linreg_dr, linreg_drr = get_dr_SKL(mean_observation, linreg_model)

        # Append to results array
        results["MISE"]["Linear regression"].append(mise)
        results["MISE Revenue"]["Linear regression"].append(mise_r)
        results["PE"]["Linear regression"].append(pe)
        results["MSFE"]["Linear regression"].append(msfe)
        results["MR"]["Linear regression"].append(mr)
        
        # Log results
        wandb.log({"MISE LinReg": mise,
                   "MISE Revenue LinReg": mise_r,
                   "PE LinReg": pe,
                   "MSFE LinReg": msfe,
                   "MR LinReg": mr},
                  step=step_id)
        
        # Log regression
        #*#*#*#*#*#*#*#*#*
        
        # Ini model
        logreg_model = LogisticRegression()

        # Train model
        logreg_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                 dataset_test=dataset_test,
                                                 model=logreg_model,
                                                 type="Classifier")

        # Get dr
        logreg_dr, logreg_drr = get_dr_SKL(mean_observation, logreg_model, type="Classifier")

        # Append to results array
        results["MISE"]["Logistic regression"].append(mise)
        results["MISE Revenue"]["Logistic regression"].append(mise_r)
        results["PE"]["Logistic regression"].append(pe)
        results["MSFE"]["Logistic regression"].append(msfe)
        results["MR"]["Logistic regression"].append(mr)
        
        # Log results
        wandb.log({"MISE LogReg": mise,
                   "MISE Revenue LogReg": mise_r,
                   "PE LogReg": pe,
                   "MSFE LogReg": msfe,
                   "MR LogReg": mr},
                  step=step_id)
        
        # Random forests
        #*#*#*#*#*#*#*#*#*
        
        # Ini model
        rf_model = RandomForestClassifier()

        # Train model
        rf_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                 dataset_test=dataset_test,
                                                 model=rf_model,
                                                 type="Classifier")

        # Get dr
        rf_dr, rf_drr = get_dr_SKL(mean_observation, rf_model, type="Classifier")

        # Append to results array
        results["MISE"]["Random forests"].append(mise)
        results["MISE Revenue"]["Random forests"].append(mise_r)
        results["PE"]["Random forests"].append(pe)
        results["MSFE"]["Random forests"].append(msfe)
        results["MR"]["Random forests"].append(mr)
        
        # Log results
        wandb.log({"MISE RF": mise,
                   "MISE Revenue RF": mise_r,
                   "PE RF": pe,
                   "MSFE RF": msfe,
                   "MR RF": mr},
                  step=step_id)
        
        # Baseline
        #*#*#*#*#*#*#*#*#*
        
        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_baseline(dataset_test=dataset_test,
                                                              num_treatments=dataset_params['num_treatments'])
        
        # Append to results array
        results["MISE"]["Baseline"].append(mise)
        results["MISE Revenue"]["Baseline"].append(mise_r)
        results["PE"]["Baseline"].append(pe)
        results["MSFE"]["Baseline"].append(msfe)
        results["MR"]["Baseline"].append(mr)
        
        # Log results
        wandb.log({"MISE Baseline": mise,
                   "MISE Revenue Baseline": mise_r,
                   "PE Baseline": pe,
                   "MSFE Baseline": msfe,
                   "MR Baseline": mr},
                  step=step_id)
        
        ###############
        # End iteration
        ###############
        
        # Save true dr curve
        true_dr, true_drr = get_true_dr(mean_observation, dataset_test)
        
        # Save treatment strengths
        treatment_strengths = get_treatment_strengths()
        
        # Plot
        figure_dr = print_summary_plot(true_dr,
                                       mlp_dr,
                                       drnets_dr,
                                       linreg_dr,
                                       logreg_dr,
                                       rf_dr,
                                       treatment_strengths,
                                       dataset_test,
                                       n,
                                       bias,
                                       "dr")
        
        figure_drr = print_summary_plot(true_drr,
                                        mlp_drr,
                                        drnets_drr,
                                        linreg_drr,
                                        logreg_drr,
                                        rf_drr,
                                        treatment_strengths,
                                        dataset_test,
                                        n,
                                        bias,
                                        "drr")
        
        # Log dr curves
        wandb.log({"DR curve": figure_dr},
                    step=step_id)
        wandb.log({"DRR curve": figure_drr},
                    step=step_id)
        
        # Increment step_id
        step_id = step_id + 1
        
#############
# Log results
#############

if (log_results_locally == True):
    # Create dicts for averages and SDs
    avgs = dict()
    for metric in metrics:
        avgs[metric] = dict()
        for experiment in experiments:
            avgs[metric][experiment] = []

    stds = dict()
    for metric in metrics:
        stds[metric] = dict()
        for experiment in experiments:
            stds[metric][experiment] = []

    for key1 in results:
        for key2 in results[key1]:
            # Calculate averages
            help = np.array(results[key1][key2])
            average = np.average(help.reshape(-1, n_iter), axis=1)
            std_dev = np.std(help.reshape(-1, n_iter), axis=1)
            # Assign averages
            avgs[key1][key2] = average
            stds[key1][key2] = std_dev

    # Save to csv
    path = "outputs/saved_results/"
    # Remove previous results
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)

    # Raw results
    MISE_raw = pd.DataFrame.from_dict(
        results["MISE"], orient="index", columns=None)
    MISE_R_raw = pd.DataFrame.from_dict(
        results["MISE_Rev"], orient="index", columns=None)
    PE_raw = pd.DataFrame.from_dict(results["PE"], orient="index", columns=None)
    MSFE_raw = pd.DataFrame.from_dict(
        results["MSFE"], orient="index", columns=None)
    MR_raw = pd.DataFrame.from_dict(results["MR"], orient="index", columns=None)

    MISE_raw.to_csv((path + "MISE_raw.csv"), sep=",", index=True)
    MISE_R_raw.to_csv((path + "MISE_Rev_raw.csv"), sep=",", index=True)
    PE_raw.to_csv((path + "PE_raw.csv"), sep=",", index=True)
    MSFE_raw.to_csv((path + "MSFE_raw.csv"), sep=",", index=True)
    MR_raw.to_csv((path + "MR_raw.csv"), sep=",", index=True)

    # Averages
    MISE_avg = pd.DataFrame.from_dict(avgs["MISE"], orient="index", columns=biases)
    MISE_R_avg = pd.DataFrame.from_dict(avgs["MISE_Rev"], orient="index", columns=biases)
    PE_avg = pd.DataFrame.from_dict(avgs["PE"], orient="index", columns=biases)
    MSFE_avg = pd.DataFrame.from_dict(avgs["MSFE"], orient="index", columns=biases)
    MR_avg = pd.DataFrame.from_dict(avgs["MR"], orient="index", columns=biases)

    MISE_avg.to_csv((path + "MISE_avg.csv"), sep=",", index=True)
    MISE_R_avg.to_csv((path + "MISE_R_avg.csv"), sep=",", index=True)
    PE_avg.to_csv((path + "PE_avg.csv"), sep=",", index=True)
    MSFE_avg.to_csv((path + "MSFE_avg.csv"), sep=",", index=True)
    MR_avg.to_csv((path + "MR_avg.csv"), sep=",", index=True)

    # Standard deviations
    MISE_std = pd.DataFrame.from_dict(stds["MISE"], orient="index", columns=biases)
    MISE_R_std = pd.DataFrame.from_dict(stds["MISE_Rev"], orient="index", columns=biases)
    PE_std = pd.DataFrame.from_dict(stds["PE"], orient="index", columns=biases)
    MSFE_std = pd.DataFrame.from_dict(stds["MSFE"], orient="index", columns=biases)
    MR_std = pd.DataFrame.from_dict(stds["MR"], orient="index", columns=biases)

    MISE_std.to_csv((path + "MISE_std.csv"), sep=",", index=True)
    MISE_R_std.to_csv((path + "MISE_R_std.csv"), sep=",", index=True)
    PE_std.to_csv((path + "PE_std.csv"), sep=",", index=True)
    MSFE_std.to_csv((path + "MSFE_std.csv"), sep=",", index=True)
    MR_std.to_csv((path + "MR_std.csv"), sep=",", index=True)
