# Logic based on "Learning Counterfactual Representations for Estimating Individual Dose-Response Curves" by Schwab et al. (2020)
# Pipeline by Christopher Bockel-Rickermann. Copyright (c) 2023

import sys
import numpy as np
from tqdm import tqdm
from scipy.integrate import romb

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer

from modules.data_prepper import get_observation_outcome

class DRNet_Model(pl.LightningModule):
    
    def __init__(self, config):
        """
        adaDRNet class
        Defines a network with x shared hiddenlayers where the network branches after the shared layers into E individual networks per interval
        across the possible dosage space
        :param dataset: DataGenerator.Data object
        :param config: dict with config infos for the network.
                       Structure:
                       - "inputSize"
                       - "numSteps"
                       - "numClasses"
                       - "learningRate"
                       - "batchSize"
                       - "logistic"
                       - "lambdaMMD"
                       - "hiddenContract"
                       - "activationType"
                       - "modelName"
        """
        # Ini the super module
        super(DRNet_Model, self).__init__()
        
        torch.manual_seed(42)
        
        # Save config
        self.config = config
        
        # Save settings
        self.model_name = config.get("modelName")
        self.learningRate = config.get("learningRate")
        self.batch_size = config.get("batchSize")
        self.num_steps = config.get("numSteps")
        self.num_layers = config.get("numLayers") # For both shared and hidden layers
        self.num_heads = config.get("numHeads") # In DRNets this is variable E
        self.input_size = config.get("inputSize")
        self.hidden_size = config.get("hiddenSize")
        
        # Save step size based on num_heads
        self.step_size = (1.0 / self.num_heads)
        
        # Initialize trainer
        self.trainer = Trainer(max_steps=self.num_steps,
                               fast_dev_run=False,
                               reload_dataloaders_every_n_epochs=False)
        
        # Structure
        # Shared layers
        self.shared_layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size)])
        self.shared_layers.extend([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers - 1)])
        
        # Hidden head layers
        self.head_layers = nn.ModuleList()
        for i in range(self.num_heads):
            help_head = nn.ModuleList([nn.Linear(self.hidden_size + 1, self.hidden_size)])
            help_head.extend([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers - 1)])
            help_head.extend([nn.Linear(self.hidden_size, 1)])
            self.head_layers.append(help_head)
            
            
    # Define forward step
    def forward(self, x, d):
        """"
        Defines the forward pass in the network
        :param x: input dat
        :param t: treatment vector
        """
        x = x.float()
        # Iterate over shared layers
        for i in range(self.num_layers):
            # Feed x to each layer one by one
            x = F.elu(self.shared_layers[i](x))
        # Reshape d
        d = d.reshape(-1,1).float()
        
        # Add t to x
        x = torch.cat((x, d), dim=1)
        
        # Determine which head to be calculated
        head_id = (torch.floor(d / self.step_size)).int()
        
        # Results array
        res = torch.zeros(x.shape[0], self.num_heads)
        
        # Iterate over head layers (+ output layer)
        for i in range(self.num_heads):
            helper_x = F.elu(self.head_layers[i][0](x))
            for j in range(1, self.num_layers):
                # Feed x to each layer one by one
                helper_x = F.elu(self.head_layers[i][j](helper_x))
            # Calc output
            helper_x = torch.sigmoid(self.head_layers[i][self.num_layers](helper_x))
            
            # Attach to res
            res[:, i] = helper_x.reshape(-1)
            
        # Return
        return res
        
        
    # Training step
    def training_step(self, batch, batch_idx):
        """
        What happens inside the training loop
        :param batch:       batch on which the network will be trained
        :param batch_idx:   batch index
        :return:            loss of the training batch forward passed through the network
        """
        
        # Get batch items
        x, y_true, d = batch
        
        # Get results of forward pass
        y = self(x, d)
        
        # Save copy
        y_help = y
        
        # Detach
        y_help = y_help.detach().clone()
        
        # Assign true value to only the head corresponding to d
        for obs in range(x.shape[0]):
            head_id = (torch.floor(d[obs] / self.step_size)).int()
            # Quick and dirty error fix for dosage d == 1
            if (head_id == self.num_heads):
                head_id = head_id - 1
            y_help[obs, head_id] = y_true[obs]
        
        # Get the loss
        loss_mse = F.mse_loss(y, y_help)
        
        return loss_mse
    
    def configure_optimizers(self):
        """
        Specifying the optimizer we want to use for training the system
        :return: the optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)
    
    def dataloader(self, dataset, shuffle=True):
        """
        Loads training data from the dataset
        :param data: Dataset
        :param shuffle: If shuffle or not
        :return: iterable training dataset
        """

        # Generate torchDataset object
        torchData = torchDataset(dataset)
        # Generate DataLoader
        loader = DataLoader(torchData, batch_size=self.batch_size, shuffle=shuffle)
        
        return loader
    
    def trainModel(self, dataset):
        """
        Train network
        """
        loader = self.dataloader(dataset)
        # Fit
        self.trainer.fit(self, loader) 

        
    def validateModel(self, dataset_val):
        """
        Get MSE on validation set
        """
        
        # Generate torchDataset object
        torchData = torchDataset(dataset_val)
        
        # Get true outcomes
        x, y_true, d = torchData.get_data()
        
        # Generate predictions
        y = self(x, d)
        
        # Save copy
        y_help = y
        
        # Detach
        y_help = y_help.detach().clone()
        
        # Assign true value to only the head corresponding to d
        for obs in range(x.shape[0]):
            head_id = (torch.floor(d[obs] / self.step_size)).int()
            # Quick and dirty error fix for dosage d == 1
            if (head_id == self.num_heads):
                head_id = head_id - 1
            y_help[obs, head_id] = y_true[obs]
        
        # Get mse
        val_mse = F.mse_loss(y, y_help).item()
        
        # Return mse multiplied by number of head
        return val_mse*self.num_heads
    
    
    def predictObservation(self, x, d):
        # Get outcomes (array form)
        outcomes = self(x, d)
        
        # Ini results array
        predictions = torch.zeros(x.shape[0])
        
        # Assign true value from correct head according to d
        for obs in range(x.shape[0]):
            head_id = (torch.floor(d[obs] / self.step_size)).int()
            # Quick and dirty error fix for dosage d == 1
            if (head_id == self.num_heads):
                head_id = head_id - 1
            predictions[obs] = outcomes[obs, head_id]
            
        predictions = predictions.detach().numpy()
            
        return predictions
        
    
    def get_dr(self, observation):
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = torch.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
        
        x = torch.from_numpy(observation).repeat(num_integration_samples, 1)
        
        dr_curve = self.predictObservation(x, treatment_strengths)
        
        drr_curve = (1 - dr_curve) * treatment_strengths.detach().numpy()
        
        return dr_curve, drr_curve
    
        
    def compute_metrics(self, dataset_test):
        # Initialize result arrays
        mises = []
        mises_R = []
        pes = []
        msfes = []
        mrs = []
        
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 0.999 / num_integration_samples
        treatment_strengths = torch.linspace(np.finfo(float).eps, 0.999, num_integration_samples)
        
        # Get number of observations
        num_test_obs = dataset_test['x'].shape[0]
        
        # Start iterating
        for obs_id in tqdm(range(num_test_obs), leave=False, desc="Evaluate test observations", ncols=125):
            observation = dataset_test["x"][obs_id]
            # Get x
            x = torch.from_numpy(observation).repeat(num_integration_samples, 1)
            
            true_outcomes = [get_observation_outcome(observation, dataset_test['metadata']['v'], 0, d) for d in treatment_strengths]
            true_outcomes = np.array(true_outcomes)
            
            pred_outcomes = np.array(self.predictObservation(x, treatment_strengths))
            
            # Calculate the revenue generated (see readme)
            pred_revenue = treatment_strengths * (1 - pred_outcomes)
            actual_revenue = treatment_strengths * (1 - true_outcomes)
            
            # Calculate MISE for revenue via romb fct and append
            mise_R = romb((pred_revenue - actual_revenue) ** 2, dx=step_size)
            mises_R.append(mise_R)
            
            # Calculate MISE for dosage curve
            mise = romb((pred_outcomes - true_outcomes) ** 2, dx=step_size)
            mises.append(mise)
            
            # Find best treatment strength
            best_pred_d = treatment_strengths[np.argmax(pred_revenue)].item()
            best_actual_d = treatment_strengths[np.argmax(actual_revenue)].item()
            
            # Get policy error by comparing best predicted and best actual dosage
            policy_error = (best_pred_d - best_actual_d) ** 2
            pes.append(policy_error)
            
            # Calculate missed_revenues
            pred_best_revenue = actual_revenue[np.argmax(pred_revenue)]
            actual_best_revenue = np.amax(actual_revenue.numpy())
            # Calculate and append
            missed_revenue = 1 - (pred_best_revenue / actual_best_revenue)
            mrs.append(missed_revenue.item())
            
            # Calculate msfe
            # Save true dose and response
            true_dose = torch.tensor(dataset_test['d'][obs_id])
            true_response = get_observation_outcome(observation, dataset_test['metadata']['v'], 0, true_dose.item()) 
            calculated_response = self.predictObservation(torch.from_numpy(observation).unsqueeze(0), true_dose.unsqueeze(0)).item()
            # Calculate and append
            sfe = (true_response - calculated_response) ** 2
            msfes.append(sfe)
            
        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(mises_R)), np.sqrt(np.mean(pes)), np.sqrt(np.mean(msfes)), np.sqrt(np.mean(mrs))            
            
        
class torchDataset(Dataset):
    def __init__(self, dataset):
        """
        Sub-class of torch dataset class
        :param data: Data object
        """
        
        # Assign values according to indices    
        self.x = torch.from_numpy(dataset["x"])
        self.y = torch.from_numpy(dataset["y"])
        self.d = torch.from_numpy(dataset["d"])
        # Save length
        self.length = dataset["x"].shape[0]
    
    # Define necessary fcts
    def get_data(self):
        return self.x, self.y, self.d
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.d[index]
    
    def __len__(self):
        return self.length