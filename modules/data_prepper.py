# Pipeline by Christopher Bockel-Rickermann. Copyright (c) 2022

from __future__ import print_function

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Create data class


class Data_object():
    def __init__(self, args):
        np.random.seed(args['seed'])

        self.num_treatments = args['num_treatments']
        self.treatment_selection_bias = args['treatment_selection_bias']
        self.dosage_selection_bias = args['dosage_selection_bias']
        self.binary_targets = args['binary_targets']

        # Set test and val fraction
        self.test_fraction = args['test_fraction']
        self.val_fraction = args['val_fraction']

        # Set noise
        self.noise_std = args['noise_std']
        
        # Set mode of assigned dosage
        self.mode = np.random.uniform(0, 0.999)

        # Safe location of dataset
        loc = args['dataset'] + '.csv'
        info_loc = args['dataset'] + '_info.csv'

        # Load dataset and dataset info from specified location
        self.data = pd.read_csv(loc, sep=",", index_col=False)
        self.data_info = pd.read_csv(info_loc, sep=",", index_col=False)

        # Transform according to _info.csv file:
        # Remove ID (allows for multiple ids)
        if 'id' in self.data_info['Variable_Type'].values:
            # For every id column
            for col in self.data_info[self.data_info['Variable_Type'].eq('id')]['Variable_Name']:
                # Drop column
                self.data.drop(col, axis=1, inplace=True)

        # Save target separately (allows for only one target)
        if 'target' in self.data_info['Variable_Type'].values:
            # For every target column
            for col in self.data_info[self.data_info['Variable_Type'].eq('target')]['Variable_Name']:
                # Drop target from dataframe
                self.data.drop(col, axis=1, inplace=True)

        # Save dosage variable (allows for one dosage)
        if 'dosage' in self.data_info['Variable_Type'].values:
            # Find target column
            col = self.data_info[self.data_info['Variable_Type'].eq('dosage')]['Variable_Name']
            # Save target
            self.dosage = self.data[col]
            # Drop target from dataframe
            self.data.drop(col, axis=1, inplace=True)
        else:
            self.dosage = np.zeros(shape=(self.data.shape[0]))

        # Check if categorical vars to be included
        if args['include_cat_vars'] == True:
            # Create dummies
            for col in self.data_info[self.data_info['Variable_Type'].eq('cat')]['Variable_Name']:
                # Dummy-fy column
                self.data = pd.concat([self.data, pd.get_dummies(self.data[col], prefix=col)], axis=1)
                # Drop original column
                self.data.drop(col, axis=1, inplace=True)
        # Else, drop categorical vars
        else:
            # For every cat var column
            for col in self.data_info[self.data_info['Variable_Type'].eq('cat')]['Variable_Name']:
                # Drop column
                self.data.drop(col, axis=1, inplace=True)

        # Make self.data np array
        self.data = self.data.to_numpy()

        # Normalize observations
        # Create new dataset with normalized values
        self.norm_data = self.normalize_data(self.data)

        # Specify number of weights
        self.num_weights = 5

        # Initialize v array (which variables impact dr curve)
        # Adds one additional dimension on top of num_weights, that is to calculate dosage selection bias
        self.v = np.zeros(shape=(self.num_treatments,
                                 self.num_weights,
                                 self.data.shape[1]))
        
        # Calculate weights
        # Also calc weights for the added index
        for i in range(self.num_treatments):
            for j in range(self.num_weights):
                self.v[i][j] = np.random.uniform(0, 1, size=(self.data.shape[1]))
                self.v[i][j] = self.v[i][j] / np.linalg.norm(self.v[i][j])
                
        # Get min and max values per np.dot(x, v[0][0:4])
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        v4 = []
        for observation in self.norm_data:
            v0.append(np.dot(observation, self.v[0][0]))
            v1.append(np.dot(observation, self.v[0][1]))
            v2.append(np.dot(observation, self.v[0][2]))
            v3.append(np.dot(observation, self.v[0][3]))
            v4.append(np.dot(observation, self.v[0][4]))
        
        # Save as array    
        self.v_min_max = np.zeros((5,2))
        self.v_min_max[0,0] = min(v0)
        self.v_min_max[0,1] = max(v0)
        self.v_min_max[1,0] = min(v1)
        self.v_min_max[1,1] = max(v1)
        self.v_min_max[2,0] = min(v2)
        self.v_min_max[2,1] = max(v2)
        self.v_min_max[3,0] = min(v3)
        self.v_min_max[3,1] = max(v3)
        self.v_min_max[4,0] = min(v4)
        self.v_min_max[4,1] = max(v4)           
                
        # Generate dataset
        self.dataset = self.generate_dataset(
            self.norm_data, self.num_treatments)

    # Define normalization function
    def normalize_data(self, features):
        # Normalize based on min/max values per variable
        x = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

        # Divide by norm of variable vector
        for i in range(x.shape[0]):
            x[i] = x[i] / np.linalg.norm(x[i])

        return x

    # Generate actual dataset for SCIGAN
    def generate_dataset(self, all_observations, num_treatments):
        # Set up dataset as dict
        dataset = dict()
        dataset['x'] = []
        dataset['y_cont'] = []
        dataset['y_binary'] = []
        dataset['t'] = []
        dataset['d'] = []
        dataset['modal_rate'] = []
        # Save true d
        dataset['d_true'] = self.dosage
        # Save metadata
        dataset['metadata'] = dict()
        dataset['metadata']['v'] = self.v
        dataset['metadata']['treatment_selection_bias'] = self.treatment_selection_bias
        dataset['metadata']['dosage_selection_bias'] = self.dosage_selection_bias
        dataset['metadata']['noise_std'] = self.noise_std

        # Iterate over all observations
        for observation in tqdm(all_observations, leave=False, desc="Generate synthetic targets", ncols=125):
            t, dosage, y, mode = generate_observation(x=observation,
                                                      v=self.v,
                                                      num_treatments=num_treatments,
                                                      treatment_selection_bias=self.treatment_selection_bias,
                                                      dosage_selection_bias=self.dosage_selection_bias,
                                                      noise_std=self.noise_std,
                                                      modal_dose=self.mode,
                                                      low=self.v_min_max[4,0],
                                                      high=self.v_min_max[4,1])
            dataset['x'].append(observation)
            dataset['t'].append(t)
            dataset['d'].append(dosage)
            dataset['y_cont'].append(y)
            dataset['modal_rate'].append(mode)
            # Transform percentage values to binary
            y_binary = float(np.random.binomial(1,y,1))
            dataset['y_binary'].append(y_binary)

        # Transform to np array
        for key in ['x', 't', 'd', 'd_true', 'y_cont', 'y_binary', 'modal_rate']:
            dataset[key] = np.array(dataset[key])

        # Save min/max values of d to assure overlap
        dataset['metadata']['d_min'] = np.min(dataset['d'])
        dataset['metadata']['d_max'] = np.max(dataset['d'])
        # Transfor d to be in [0,1]
        dataset['d'] = (dataset['d'] - dataset['metadata']['d_min']) / (dataset['metadata']['d_max'] - dataset['metadata']['d_min'])

        # Save true dose as np array
        dataset['d_true'] = np.array(dataset['d_true'])

        train_indices, val_indices, test_indices = get_split_indices(num_observations=dataset['x'].shape[0],
                                                                     observations=dataset['x'],
                                                                     treatments=dataset['t'],
                                                                     test_fraction=self.test_fraction,
                                                                     val_fraction=self.val_fraction)

        dataset['metadata']['train_index'] = train_indices
        dataset['metadata']['val_index'] = val_indices
        dataset['metadata']['test_index'] = test_indices
        
        # Save y as binary or continuous based on parameters
        if self.binary_targets == True:
            dataset['y'] = dataset['y_binary']
        else:
            dataset['y'] = dataset['y_cont']

        return dataset


# Define function for creating split indices
def get_split_indices(num_observations, observations, treatments, val_fraction, test_fraction):
    # Number of val observations
    num_val_observations = int(np.floor(num_observations * val_fraction))
    # Number of test observations
    num_test_observations = int(np.floor(num_observations * test_fraction))
    
    if val_fraction > 0:
        # Initialize StratShuffleSplit
        test_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=num_test_observations, random_state=0)
        # Split
        rest_indices, test_indices = next(
            test_sss.split(observations, treatments))

        # Initialize StratShuffleSplit
        val_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=num_val_observations, random_state=0)
        # Split
        train_indices, val_indices = next(val_sss.split(
            observations[rest_indices], treatments[rest_indices]))

        return train_indices, val_indices, test_indices
    
    else:
        # Initialize StratShuffleSplit
        test_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=num_test_observations, random_state=0)
        # Split
        rest_indices, test_indices = next(
            test_sss.split(observations, treatments))

        return rest_indices, None, test_indices


# Define function to create train and test data from data object
def get_dataset_splits(dataset):
    dataset_keys = ['x',
                    't',
                    'd',
                    'y_cont',
                    'd_true',
                    'y_binary',
                    'y',
                    'modal_rate']

    train_index = dataset['metadata']['train_index']
    test_index = dataset['metadata']['test_index']
    val_index = dataset['metadata']['val_index']

    dataset_train = dict()
    dataset_val = dict()
    dataset_test = dict()
    for key in dataset_keys:
        dataset_train[key] = dataset[key][train_index]
        dataset_val[key] = dataset[key][val_index]
        dataset_test[key] = dataset[key][test_index]

    dataset_train['metadata'] = dataset['metadata']
    dataset_val['metadata'] = dataset['metadata']
    dataset_test['metadata'] = dataset['metadata']

    return dataset_train, dataset_val, dataset_test


# Define function to generate an observation
def generate_observation(x, v, num_treatments, treatment_selection_bias, dosage_selection_bias, noise_std, modal_dose, low, high):
    outcomes = []
    dosages = []
    modes = []
    
    # Assign dosage based on treatment and treatment selection bias
    for treatment in range(num_treatments):
        if (treatment == 0):
            # Calculate linear combination of inputs and standardize
            low = low - 0.0001
            high = high + 0.0001
            lin_comb = (np.dot(x, v[0][4]) - low) / (high - low)
            """ OPTION 1: Modal value is linear combination of inputs
            """
            # Calculate strength of the selection bias
            alpha = 1 + dosage_selection_bias
            # Calculate modal value of beta distr. as linear combination of inputs
            mode = lin_comb
            
            # Save beta
            beta = compute_beta(alpha, modal_rate=mode)
            # Calculate the dosage distribution
            dosage = np.random.beta(alpha, beta)
            y = get_observation_outcome(x, v, treatment, dosage)
            """
            END OPTION 1"""
            
            """ OPTION 2: Selection bias is linear combination of inputs
            
            # Calculate strength of the selection bias
            alpha = 1 + dosage_selection_bias * lin_comb
            # Calculate modal value of beta distr. as linear combination of inputs
            mode = modal_dose
            
            # Save beta
            beta = compute_beta(alpha, modal_rate=mode)
            # Calculate the dosage distribution
            dosage = np.random.beta(alpha, beta)
            y = get_observation_outcome(x, v, treatment, dosage)
            
            END OPTION 2"""
            

        # Add other treatments for num_treatments > 1

        outcomes.append(y)
        dosages.append(dosage)
        modes.append(mode)

    treatment_coeff = [treatment_selection_bias *
                       (outcomes[i] / np.max(outcomes)) for i in range(num_treatments)]

    treatment = np.random.choice(num_treatments, p=softmax(treatment_coeff))

    # Assign an additional error specified by 'noise_std' variable and bind to [0,1]
    return treatment, dosages[treatment], np.clip(a=(outcomes[treatment] + np.random.normal(0, noise_std)), a_min=0, a_max=1), modes[treatment]


# Generate function to calculate softmax
def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


# Generate function that returns y-value
def get_observation_outcome(x, v, treatment, dosage):
    if (treatment == 0):
        
        y = generalized_logistic(dosage=dosage,
                                 input1=np.dot(x, v[treatment][0]),
                                 input2=np.dot(x, v[treatment][1]),
                                 input3=np.dot(x, v[treatment][2]),
                                 input4=np.dot(x, v[treatment][3]))
        """
        y = double_logistic(dosage=dosage,
                            input1=np.dot(x, v[treatment][0]),
                            input2=np.dot(x, v[treatment][1]),
                            input3=np.dot(x, v[treatment][2]),
                            input4=np.dot(x, v[treatment][3]))
        """
        
    # Add additional treatments when num_treatments > 1

    return y

# Implement dose response function
def generalized_logistic(dosage, input1, input2, input3, input4):
    """
    Function type: Generalized logistic function
    Parameters: (Ones modelled variable are marked with "*")
        A*: Lower left asymptote 
        K*: Upper right asymptote when C=1. If A=0 and C=1 then K is called the carrying capacity
            If A is > than calculated value, then take A
        B:  Growth rate or steepness
        NU: Affects near which asymptote maximum growth occurs. NU >! 0. For NU < 1, shift to the right, left otherwise
        Q:  Related to the value of Y(0)
        C:  Typically takes a value of 1. Otherwise the upper asymptote is (A + (K-A)/(C^(1/U))
        P:  Reposition of "center" of the curve on the x-axis
    """
    
    A = float(0.2*(input1))
    K = float(0.8 + 0.2*(input2))
    B = float(0.5 + 5*input3)
    NU = float(1)
    Q = float(1) 
    C = float(1)
    P = float(input4)
    
    response = (A + ((K - A)/((C + Q * np.exp(-B*(10*(dosage-P))))**(1/NU))))
    return response

def logistic(dosage):
    """
    Function type: Generalized logistic function
    Parameters: (Ones modelled variable are marked with "*")
        A*: Lower left asymptote 
        K*: Upper right asymptote when C=1. If A=0 and C=1 then K is called the carrying capacity
            If A is > than calculated value, then take A
        B:  Growth rate or steepness
        NU: Affects near which asymptote maximum growth occurs. NU >! 0. For NU < 1, shift to the right, left otherwise
        Q:  Related to the value of Y(0)
        C:  Typically takes a value of 1. Otherwise the upper asymptote is (A + (K-A)/(C^(1/U))
        P:  Reposition of "center" of the curve on the x-axis
    """
    
    A = float(0)
    K = float(1)
    B = float(2)
    NU = float(1)
    Q = float(1) 
    C = float(1)
    P = float(0.5)
    
    response = (A + ((K - A)/((C + Q * np.exp(-B*(10*(dosage-P))))**(1/NU))))
    return response

# Implement a function of two stacked logistic curves
def double_logistic(dosage, input1, input2, input3, input4):
    """
    Function type: Generalized logistic function
    Parameters: (Ones modelled variable are marked with "*")
        A: Base-rate
        B: Beginning of the plateau (with respect to p)
        C: End of the plateau (with respect to p)
        D: Hight of the plateau (with respect to Y)
    """
    
    A = float(input1)
    B = float(input2)
    C = float(input3)
    D = float(input4)
    
    response = ((0.2 * A) + # Base
                ((0.8 * B) * logistic((dosage / C))) + # First sigmoid
                ((1 - 0.2 * A - 0.8 * B) * logistic(((dosage - D) / (1 - D)))) # Second sigmoid
                )
    
    return response

# Beta computation
# Goal is to assign on average the mode of the observed data
# From Bica (2020)
def compute_beta(alpha, modal_rate):
    beta = (alpha - 1.0) / float(modal_rate) + (2.0 - alpha)
    return beta