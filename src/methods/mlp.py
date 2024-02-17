# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

# Load necessary modules
import numpy as np
from tqdm import tqdm

from scipy.integrate import romb

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.data.datagen import get_outcome

from src.utils.torch import torchDataset

class MLP(pl.LightningModule):
    def __init__(self, config):
        # Ini the super module
        super(MLP, self).__init__()
        
        torch.manual_seed(42)
        
        # Save config
        self.config = config
        
        # Save settings
        self.learningRate = config.get('learningRate')
        self.batch_size = config.get('batchSize')
        self.num_steps = config.get('numSteps')
        self.num_layers = config.get('numLayers')
        self.input_size = config.get('inputSize')
        self.hidden_size = config.get('hiddenSize')
        
        # Initialize trainer
        self.trainer = Trainer(max_steps=self.num_steps,
                               max_epochs=9999, # To not interupt step-wise iteration
                               fast_dev_run=False,
                               reload_dataloaders_every_n_epochs=False,
                               enable_progress_bar=False,
                               enable_checkpointing=False,
                               enable_model_summary=False,
                               logger=False,
                               accelerator='cpu',
                               devices=1)
        
        # Structure
        # Initialize shared layers
        self.layers = nn.Sequential(nn.Linear(self.input_size + 1, self.hidden_size))
        self.layers.append(nn.ELU())
        # Add additional layers
        for i in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(nn.ELU())
        # Add output layer
        self.layers.append(nn.Linear(self.hidden_size, 1))
        self.layers.append(nn.Sigmoid())        
                       
    # Define forward step
    def forward(self, x, d):
        # Reshape d
        d = d.reshape(-1,1)
        
        # Add t to x
        x = torch.cat((x, d), dim=1)
        
        # Feed through layers
        x = self.layers(x)
            
        # Return
        return x
    
    # Training step
    def training_step(self, batch, batch_idx):
        # Get batch items
        x, y_true, d = batch
        
        # Get results of forward pass
        y = self(x, d).squeeze()
        
        # Get the loss
        loss_mse = F.mse_loss(y, y_true)
        
        return loss_mse
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)
    
    def dataloader(self, dataset, shuffle=True):
        torch_data = torchDataset(dataset)
        # Generate DataLoader
        return DataLoader(torch_data, batch_size=self.batch_size, shuffle=shuffle)
    
    def trainModel(self, dataset):
        # Define loader
        loader = self.dataloader(dataset)
        # Fit
        self.trainer.fit(self, loader) 
        
    def validateModel(self, dataset):
        torch_data = torchDataset(dataset)
        # Get true outcomes
        x, y_true, d = torch_data.get_data()
        
        # Generate predictions
        y = self(x,d).squeeze()
        
        # Get mse
        val_mse = F.mse_loss(y, y_true).detach().item()
        
        # Return mse
        return val_mse
    
    def predictObservation(self, x, d):
        # Get outcomes (array form)
        outcomes = self(x, d).detach().numpy()
            
        return outcomes
    
    def getDR(self, observation):
        # Save observation as torch tensor
        observation = torch.Tensor(observation)
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        x = observation.repeat(num_integration_samples, 1)
        
        dr_curve = self.predictObservation(x, treatment_strengths).squeeze()
        
        return dr_curve
    
    def computeMetrics(self, dataset):
        torch_data = torchDataset(dataset)
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
        treatment_strengths = torch.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get number of observations
        num_test_obs = torch_data.x.shape[0]
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc='Evaluate test observations'):
            # Get observation
            observation = torch_data.x[obs_id]
            # Repeat observation to get x
            x = observation.repeat(num_integration_samples, 1)
            
            true_outcomes = [get_outcome(observation, d , torch_data.v, torch_data.response) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            # Predict outcomes
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths)).reshape(1,-1).squeeze()
            
            ## MISE ##
            mise = romb((pred_outcomes - true_outcomes) ** 2, dx=step_size)
            mises.append(mise)
            
            ## MISE R ##
            true_rev = (1 - true_outcomes) * treatment_strengths.numpy()
            pred_rev = (1 - pred_outcomes) * treatment_strengths.numpy()
            mise_r = romb((pred_rev - true_rev) ** 2, dx=step_size)
            mises_r.append(mise_r)
            
            ## PE ##
            best_pred_d = treatment_strengths[np.argmax(pred_rev)].detach()
            best_actual_d = treatment_strengths[np.argmax(true_rev)].detach()
            pe = (best_pred_d - best_actual_d) ** 2
            pes.append(pe)
            
            ## MR ##
            pred_best_revenue = true_rev[np.argmax(pred_rev)]
            actual_best_revenue = np.amax(true_rev)
            mr = 1 - (pred_best_revenue / actual_best_revenue)
            mrs.append(mr)
            
            ## BS ##
            fact_outcome = torch_data.y[obs_id]
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
            
            bs = ((fact_outcome - pred_outcome) ** 2).detach().item()
            bss.append(bs)
            
        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(mises_r)), np.sqrt(np.mean(pes)), np.sqrt(np.mean(mrs)), np.sqrt(np.mean(bss))