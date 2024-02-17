# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

import torch

from src.data.datagen import get_outcome

class Baseline():
    def __init__(self, config):
        np.random.seed(42)
        
        # Save settings
        self.dummy = config.get('dummy')
        
    def trainModel(self, dataset):
        # Do nothing
        nothing = 1+1
        
    def validateModel(self, dataset):
        # Do nothing
        
        return 99.99
        
    def predictObservation(self, x, d):
        num_obs = x.shape[0]
        
        outcomes = np.array([99.99 for n in range(num_obs)])
        
        return outcomes
        
    def getDR(self, observation):
        # Save observation as torch tensor
        observation = torch.Tensor(observation)
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Repeat observation
        x = observation.repeat(num_integration_samples, 1).numpy()
        
        # Predict dr curve
        dr_curve = self.predictObservation(x, treatment_strengths)
        
        return dr_curve
    
    def computeMetrics(self, dataset):
        # Initialize result arrays
        mises = []
        mises_r = []
        pes = []
        mrs = []
        bss = []
        
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1 / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get number of observations
        num_test_obs = dataset['x'].shape[0]
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='Evaluate test observations'):
            # Get observation
            observation = dataset['x'][obs_id]
            observation = torch.Tensor(observation)
            # Repeat observation to get x
            x = observation.repeat(num_integration_samples, 1).numpy()
            
            true_outcomes = [get_outcome(observation, d , dataset['v'], dataset['gt']) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Predict outcomes
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths))
            
            ## MISE ##
            mise = 99.99 ** 2
            mises.append(mise)
            
            ## MISE R ##
            true_rev = (1 - true_outcomes) * treatment_strengths
            pred_rev = (1 - pred_outcomes) * treatment_strengths
            mise_r = 99.99 ** 2
            mises_r.append(mise_r)
            
            ## PE ##
            best_pred_d = dataset['d'][obs_id]
            best_actual_d = treatment_strengths[np.argmax(true_rev)]
            pe = (best_pred_d - best_actual_d) ** 2
            pes.append(pe)
            
            ## MR ##
            pred_best_revenue = (1 - get_outcome(observation, dataset['d'][obs_id], dataset['v'], dataset['gt'])) * dataset['d'][obs_id]
            actual_best_revenue = np.amax(true_rev)
            mr = 1 - (pred_best_revenue / actual_best_revenue)
            mrs.append(mr)
            
            ## BS ##
            bs = 99.99 ** 2
            bss.append(bs)
        
        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(mises_r)), np.sqrt(np.mean(pes)), np.sqrt(np.mean(mrs)), np.sqrt(np.mean(bss))
    