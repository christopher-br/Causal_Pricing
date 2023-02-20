# SCIGAN implementation based on code by Ioana Bica (2020)
# Remaining pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022

# Preliminaries
import os
# Change verbosity in os variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import tensorflow as tf
# Avoid tf to print on std errors (e.g., compatibility)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load data class, SCIGAN and SCIMLP
from modules.MLP import SCIMLP_Model
from modules.SCIGAN import SCIGAN_Model
from modules.DRNets import DRNets_Model
from modules.data_prepper import Data_object, get_dataset_splits

# Load benchmark models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

# Load eval and print utils
from utils.eval_utils import compute_metrics_SCI, compute_metrics_SKL, compute_metrics_baseline

# Load remaining modules
from tqdm import tqdm
import numpy as np
import pandas as pd

# Save experiments and metrics
experiments = ["SCIGAN", 
               "SCIMLP", 
               "DRNet", 
               "Linear regression",
               "Logistic regression", 
               "Random forest",
               "Baseline"]

metrics = ["MISE", 
           "MISE_Rev", 
           "PE", 
           "MSFE", 
           "MR"]

# Initialize results dict
results = dict()
for metric in metrics:
    results[metric] = dict()
    for experiment in experiments:
        results[metric][experiment] = []

# Number of iterations
n_iter = 10
biases = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]

# Counter
step_id = 0

# Iterate over iterations and biases
for bias in tqdm(biases, desc="Iterating over biases", ncols=125):
    for n in tqdm(range(n_iter), leave=False, desc="Iteration", ncols=125):

        # Save dataset parameters
        dataset_params = dict()
        dataset_params['num_treatments'] = int(1)
        dataset_params['treatment_selection_bias'] = float(0)
        dataset_params['dosage_selection_bias'] = float(bias)
        dataset_params['test_fraction'] = float(0.10)
        dataset_params['val_fraction'] = float(0.10)
        # Added args
        dataset_params['dataset'] = str("datasets/...") # To be specified
        dataset_params['include_cat_vars'] = True
        dataset_params['binary_targets'] = True
        dataset_params['noise_std'] = float(0.1)
        dataset_params['seed'] = 10*n + 1

        # Initialize data
        data_class = Data_object(dataset_params)
        dataset = data_class.dataset
        dataset_train, dataset_val, dataset_test = get_dataset_splits(dataset)
        num_features = dataset_train['x'].shape[1]

        # Calculate average observation
        mean_observation = np.mean(dataset_test['x'], axis=0)

        # SCIGAN
        scigan_model_dir = 'outputs/saved_models/SCIGAN'
        if os.path.exists(scigan_model_dir):
            shutil.rmtree(scigan_model_dir)

        scigan_params = {'num_treatments': 1,
                         'num_features': num_features,
                         'batch_size': 128,
                         'num_dosage_samples': 5,
                         'alpha': 1,
                         'h_inv_eqv_dim': 128,
                         'h_dim': 128,
                         'num_iter_generator': 5000,
                         'num_iter_inference': 10000,
                         'export_model_dir': scigan_model_dir}

        # Ini model
        scigan_model = SCIGAN_Model(scigan_params)

        # Train model
        scigan_model.tune(Train_X=dataset_train["x"],
                          Train_T=dataset_train["t"],
                          Train_D=dataset_train["d"],
                          Train_Y=dataset_train["y"],
                          dataset_val=dataset_val,
                          batch_sizes=[128],
                          h_dims=[32, 64],
                          nums_dosage_samples=[5])
        
        # Update params
        scigan_params['batch_size'] = scigan_model.batch_size
        scigan_params['h_dim'] = scigan_model.h_dim
        scigan_params['h_inv_eqv_dim'] = scigan_model.h_inv_eqv_dim
        scigan_params['num_dosage_samples'] = scigan_model.num_dosage_samples

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SCI(dataset_test=dataset_test,
                                                 num_treatments=dataset_params['num_treatments'],
                                                 num_dosage_samples=scigan_params['num_dosage_samples'],
                                                 model_folder=scigan_model_dir)
        
        # Delete model
        del scigan_model

        # Append to results array
        results["MISE"]["SCIGAN"].append(mise)
        results["MISE_Rev"]["SCIGAN"].append(mise_r)
        results["PE"]["SCIGAN"].append(pe)
        results["MSFE"]["SCIGAN"].append(msfe)
        results["MR"]["SCIGAN"].append(mr)

        # SCIMLP
        SCIMLP_model_dir = 'outputs/saved_models/SCIMLP'
        if os.path.exists(SCIMLP_model_dir):
            shutil.rmtree(SCIMLP_model_dir)

        scimlp_params = {'num_treatments': 1,
                         'num_features': num_features,
                         'batch_size': 1024,
                         'num_dosage_samples': 1,
                         'alpha': 1,
                         'h_dim': 128,
                         'num_iter_inference': 10000,
                         'export_model_dir': SCIMLP_model_dir}

        # Ini model
        scimlp_model = SCIMLP_Model(scimlp_params)

        # Train model
        scimlp_model.tune(Train_X=dataset_train["x"],
                          Train_T=dataset_train["t"],
                          Train_D=dataset_train["d"],
                          Train_Y=dataset_train["y"],
                          dataset_val=dataset_val,
                          batch_sizes=[64, 128],
                          h_dims=[32, 64])
        
        # Update params
        scimlp_params['batch_size'] = scimlp_model.batch_size
        scimlp_params['h_dim'] = scimlp_model.h_dim
        scimlp_params['h_inv_eqv_dim'] = scimlp_model.h_inv_eqv_dim

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SCI(dataset_test=dataset_test,
                                                 num_treatments=dataset_params['num_treatments'],
                                                 num_dosage_samples=scimlp_params['num_dosage_samples'],
                                                 model_folder=SCIMLP_model_dir)
        
        # Delete model
        del scimlp_model
        
        # Append to results array
        results["MISE"]["SCIMLP"].append(mise)
        results["MISE_Rev"]["SCIMLP"].append(mise_r)
        results["PE"]["SCIMLP"].append(pe)
        results["MSFE"]["SCIMLP"].append(msfe)
        results["MR"]["SCIMLP"].append(mr)
        
        # DRNet
        # Ini val error
        drnets_val_error = np.inf
        
        # Tune
        for b_size in [64, 128, 256]:
            for n_heads in [10, 15, 20]:
                for l_rate in [0.01, 0.1]:
                    for h_size in [12, 24]:
                        for n_epochs in [2]:
                            for n_layers in [2,3]:                                
                                # Save config file
                                config = dict()
                                config["modelName"] = "DRNet"
                                config["learningRate"] = l_rate
                                config["batchSize"] = b_size
                                config["hiddenSize"] = h_size
                                config["numEpochs"] = n_epochs
                                config["numLayers"] = n_layers
                                config["numHeads"] = n_heads
                                config["inputSize"] = num_features
                                
                                # Set up model
                                dr_model_temp = DRNets_Model(config)
                                # Train
                                dr_model_temp.trainModel(dataset_train)
                                # Get validation error
                                temp_error = dr_model_temp.validateModel(dataset_test)
                                # Save new lowest error and save model
                                if (temp_error < drnets_val_error):
                                    drnets_val_error = temp_error
                                    drnets_model = dr_model_temp
        
        # Evaluate
        mise, mise_r, pe, msfe, mr = drnets_model.compute_metrics(dataset_test)
        
        # Del previous models
        del drnets_model
        # Delete temp
        del dr_model_temp
        
        # Append to results array
        results["MISE"]["DRNet"].append(mise)
        results["MISE_Rev"]["DRNet"].append(mise_r)
        results["PE"]["DRNet"].append(pe)
        results["MSFE"]["DRNet"].append(msfe)
        results["MR"]["DRNet"].append(mr)   

        # For remaining methods: Build training and test data (append dosage to x)
        # Train
        Train_X = np.column_stack((dataset_train["x"],
                                   dataset_train["d"].reshape((-1, 1))))
        Train_Y = dataset_train["y"]
        # Test
        Test_X = np.column_stack((dataset_test["x"],
                                  dataset_test["d"].reshape((-1, 1))))
        Test_Y = dataset_test["y"]

        # Linear regression

        # Ini model
        linreg_model = LinearRegression()

        # Train model
        linreg_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                 dataset_test=dataset_test,
                                                 model=linreg_model)

        # Append to results array
        results["MISE"]["Linear regression"].append(mise)
        results["MISE_Rev"]["Linear regression"].append(mise_r)
        results["PE"]["Linear regression"].append(pe)
        results["MSFE"]["Linear regression"].append(msfe)
        results["MR"]["Linear regression"].append(mr)

        # Logistic regression

        # Ini model
        logreg_model = LogisticRegression()

        # Train model
        logreg_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                 dataset_test=dataset_test,
                                                 model=logreg_model,
                                                 type="Classifier")

        # Append to results array
        results["MISE"]["Logistic regression"].append(mise)
        results["MISE_Rev"]["Logistic regression"].append(mise_r)
        results["PE"]["Logistic regression"].append(pe)
        results["MSFE"]["Logistic regression"].append(msfe)
        results["MR"]["Logistic regression"].append(mr)

        # Random forests

        # Ini model
        rf_model = RandomForestClassifier()

        # Train model
        rf_model.fit(Train_X, Train_Y)

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_SKL(Test_X=Test_X,
                                                 dataset_test=dataset_test,
                                                 model=rf_model,
                                                 type="Classifier")

        # Append to results array
        results["MISE"]["Random forest"].append(mise)
        results["MISE_Rev"]["Random forest"].append(mise_r)
        results["PE"]["Random forest"].append(pe)
        results["MSFE"]["Random forest"].append(msfe)
        results["MR"]["Random forest"].append(mr)

        # Baseline

        # Evaluate
        mise, mise_r, pe, msfe, mr = compute_metrics_baseline(dataset_test=dataset_test,
                                                      num_treatments=dataset_params['num_treatments'])

        # Append to results array
        results["MISE"]["Baseline"].append(mise)
        results["MISE_Rev"]["Baseline"].append(mise_r)
        results["PE"]["Baseline"].append(pe)
        results["MSFE"]["Baseline"].append(msfe)
        results["MR"]["Baseline"].append(mr)

        # Increment step_id
        step_id = step_id + 1

# Save results

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