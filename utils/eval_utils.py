# Pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022

import numpy as np
from scipy.integrate import romb
from tqdm import tqdm

from modules.data_prepper import get_observation_outcome


def get_true_dose_response_curve(dataset, observation, treatment_idx):
    def true_dose_response_curve(dosage):
        y = get_observation_outcome(observation, dataset['metadata']['v'], treatment_idx, dosage)
        return y

    return true_dose_response_curve


def compute_metrics_SKL(Test_X, dataset_test, model, type="Regressor"):
    treatment_idx = 0 # Only one treatment
    mises = []
    mises_R = []
    policy_errors = []
    # (Mean) squared factual error
    sfes = []
    # Missed revenues
    missed_revenues = []
    
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 0.999 / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)

    # Get the number of observations to iterate over
    num_test_obs = Test_X.shape[0]
    # Start iterating
    for obs_id in tqdm(range(num_test_obs), leave=False, desc="Evaluate test observations", ncols=125):
        # Get the observation
        observation = Test_X[obs_id,:-1] # Remove dosage
        test_data = dict()
        test_data['x'] = np.repeat(np.expand_dims(observation, axis=0), num_integration_samples, axis=0) # Attach dosages
        test_data['t'] = np.repeat(0, num_integration_samples)
        test_data['d'] = treatment_strengths
        test_data['xd'] = np.column_stack((test_data['x'], test_data['d']))
        
        # Get predicted dose response curve
        if type=="Regressor":
            pred_dose_response = model.predict(test_data['xd'])
        elif type=="Classifier":
            pred_dose_response = model.predict_proba(test_data['xd'])[:,1]
            
        pred_dose_response = np.array(pred_dose_response)
        
        # Save true dose response
        true_dose_response = [get_observation_outcome(observation, dataset_test['metadata']['v'], treatment_idx, d) for d in treatment_strengths]
        true_dose_response = np.array(true_dose_response)
        
        # Calculate the revenue generated (see readme)
        pred_revenue = treatment_strengths * (1 - pred_dose_response)
        actual_revenue = treatment_strengths * (1 - true_dose_response)
        
        # Calculate MISE for revenue via romb fct and append
        mise_R = romb((pred_revenue - actual_revenue) ** 2, dx=step_size)
        mises_R.append(mise_R)
            
        # Calculate MISE for dosage curve
        mise = romb((pred_dose_response - true_dose_response) ** 2, dx=step_size)
        mises.append(mise)
        
        # Find best treatment strength
        best_pred_d = treatment_strengths[np.argmax(pred_revenue)]
        best_actual_d = treatment_strengths[np.argmax(actual_revenue)]
        
        # Get policy error by comparing best predicted and best actual dosage
        policy_error = (best_pred_d - best_actual_d) ** 2
        policy_errors.append(policy_error)
        
        # Calculate missed_revenues
        pred_best_revenue = actual_revenue[np.argmax(pred_revenue)]
        actual_best_revenue = np.amax(actual_revenue)
        # Calculate and append
        missed_revenue = 1 - (pred_best_revenue / actual_best_revenue)
        missed_revenues.append(missed_revenue)
        
        # Calculate msfe
        # Save true dose and response
        true_dose = dataset_test['d'][obs_id]
        true_response = get_observation_outcome(observation, dataset_test['metadata']['v'], treatment_idx, true_dose) 
        # Get real observation
        obs_help = np.append(dataset_test['x'][obs_id], dataset_test['d'][obs_id])
        
        # Calculate response
        if type=="Regressor":
            calculated_response = model.predict(obs_help.reshape(1,-1))
        elif type=="Classifier":
            calculated_response = model.predict_proba(obs_help.reshape(1,-1))
            
        # Calculate and append
        sfe = (true_response - calculated_response) ** 2
        sfes.append(sfe)

    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(mises_R)), np.sqrt(np.mean(policy_errors)), np.sqrt(np.mean(sfes)), np.sqrt(np.mean(missed_revenues))

def compute_metrics_baseline(dataset_test, num_treatments):
    mises = []
    policy_errors = []
    # (Mean) squared factual error
    sfes = []
    # Missed revenues
    missed_revenues = []
    
    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 0.999 / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
    
    # Get the number of observations to iterate over
    num_test_obs = dataset_test['x'].shape[0]
    # Start iterating
    for obs_id in tqdm(range(num_test_obs), leave=False, desc="Evaluate test observations", ncols=125):
        # Get the observation
        observation = dataset_test['x'][obs_id]
        factual_dosage = dataset_test['d'][obs_id]
        for treatment_idx in range(num_treatments):
            test_data = dict()
            test_data['x'] = np.repeat(np.expand_dims(observation, axis=0), num_integration_samples, axis=0)
            test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
            test_data['d'] = treatment_strengths
            
            # Save true outcomes
            true_outcomes = [get_observation_outcome(observation, dataset_test['metadata']['v'], treatment_idx, d) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Calculate the revenue generated (see readme)
            actual_revenue = treatment_strengths * (1 - true_outcomes)
            
            # No MISE calculation as no dose response predicted
            
            # Find best treatment strength
            best_actual_d = treatment_strengths[np.argmax(actual_revenue)]
            
            # Get policy error by comparing best predicted and best actual dosage
            policy_error = (factual_dosage - best_actual_d) ** 2
            policy_errors.append(policy_error)
            
            # Calculate missed_revenues
            factual_revenue = factual_dosage * (1 - get_observation_outcome(observation, dataset_test['metadata']['v'], treatment_idx, factual_dosage))
            actual_best_revenue = np.amax(actual_revenue)
            # Calculate and append
            missed_revenue = 1 - (factual_revenue / actual_best_revenue)
            missed_revenues.append(missed_revenue)
            
            # No MSFE, as no prediction

    return 99, 99, np.sqrt(np.mean(policy_errors)), 99, np.sqrt(np.mean(missed_revenues))