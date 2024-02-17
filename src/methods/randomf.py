# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

from sklearn.ensemble import RandomForestClassifier

import torch

from src.data.datagen import get_outcome

class RandomF():
    def __init__(self, config):
        np.random.seed(42)
        
        # Save settings
        self.criterion = config.get('criterion')
        self.n_estimators = config.get('numEstimators')
        self.max_depth = config.get('maxDepth')
        
        # Ini model
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            criterion=self.criterion,
                                            max_depth=self.max_depth,
                                            random_state=42)
        
    def trainModel(self, dataset):
        # Define arrays
        train_x = np.column_stack((dataset['x'],
                                   dataset['d'].reshape((-1,1))))
        train_y = dataset['y']
        
        # Train
        self.model.fit(train_x, train_y)
        
    def validateModel(self, dataset):
        # Define arrays
        val_x = np.column_stack((dataset['x'],
                                 dataset['d'].reshape((-1,1))))
        val_y = dataset['y']
        
        preds = self.model.predict_proba(val_x)[:,1]
        
        val_mse = np.mean((preds - val_y) ** 2)
        
        return val_mse
        
    def predictObservation(self, x, d):
        obs = np.column_stack((x,d))
        
        outcomes = self.model.predict_proba(obs)[:,1]
        
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
        # Define arrays
        test_x = np.column_stack((dataset['x'],
                                 dataset['d'].reshape((-1,1))))
        test_y = dataset['y']
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
        num_test_obs = test_x.shape[0]
        
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
            mise = romb((pred_outcomes - true_outcomes) ** 2, dx=step_size)
            mises.append(mise)
            
            ## MISE R ##
            true_rev = (1 - true_outcomes) * treatment_strengths
            pred_rev = (1 - pred_outcomes) * treatment_strengths
            mise_r = romb((pred_rev - true_rev) ** 2, dx=step_size)
            mises_r.append(mise_r)
            
            ## PE ##
            best_pred_d = treatment_strengths[np.argmax(pred_rev)]
            best_actual_d = treatment_strengths[np.argmax(true_rev)]
            pe = (best_pred_d - best_actual_d) ** 2
            pes.append(pe)
            
            ## MR ##
            pred_best_revenue = true_rev[np.argmax(pred_rev)]
            actual_best_revenue = np.amax(true_rev)
            mr = 1 - (pred_best_revenue / actual_best_revenue)
            mrs.append(mr)
            
            ## BS ##
            fact_outcome = test_y[obs_id]
            # Find closest dosage in treatment strengths
            fact_d = dataset['d'][obs_id]
            lower_id = np.searchsorted(treatment_strengths, fact_d, side="left") - 1
            upper_id = lower_id + 1
            
            lower_d = treatment_strengths[lower_id]
            upper_d = treatment_strengths[upper_id]
            
            lower_est = pred_outcomes[lower_id]
            upper_est = pred_outcomes[upper_id]
            
            # Get calc as linear interpolation
            pred_outcome = ((fact_d - lower_d) * upper_est + (upper_d - fact_d) * lower_est) / (upper_d - lower_d)
            
            bs = ((fact_outcome - pred_outcome) ** 2)
            bss.append(bs)
        
        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(mises_r)), np.sqrt(np.mean(pes)), np.sqrt(np.mean(mrs)), np.sqrt(np.mean(bss))
    