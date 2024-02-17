# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# Create data object
class Data_object():
    def __init__(self, args):
        '''
        Initialize data object. Takes a dictionary of arguments for generating the data.
        
        args (dict):
        'confounding_bias' (float, >=0 OR -1): The strength of the confounding bias. 0 if no bias. -1 for using observed bids
        'test_fraction' (float, [0,1]): The fraction of observations used for testing.
        'val_fraction' (float, [0,1]): The fraction of observations used for validation.
        'noise_std' (float): The standard deviation of assigned errors
        'dataset' (str): The name of the dataset without file endings
        'gt' (str): The ground truth, either 'ss' or 'rc'
        'seed' (int): The random seed for data generation
        '''
        
        # Set seed
        np.random.seed(args['seed'])
        
        ###############
        # Save params #
        ###############
        
        self.confounding_bias = args['confounding_bias']

        # Set test and val fraction
        self.test_fraction = args['test_fraction']
        self.val_fraction = args['val_fraction']

        # Set noise
        self.noise_std = args['noise_std']
        
        # Set gt
        self.gt = args['gt']

        # Safe location of dataset
        loc = args['dataset'] + '.csv'
        info_loc = args['dataset'] + '_info.csv'

        # Load dataset and dataset info from specified location
        self.data = pd.read_csv(loc, sep=",", index_col=False)
        self.data_info = pd.read_csv(info_loc, sep=",", index_col=False)
        
        ##########################################
        # Transform according to _info.csv file: #
        ##########################################
        
        # 1. Remove ID
        if 'id' in self.data_info['Variable_Type'].values:
            # For every id column
            for col in self.data_info[self.data_info['Variable_Type'].eq('id')]['Variable_Name']:
                # Drop column
                self.data.drop(col, axis=1, inplace=True)
                
        # 2. Remove target
        if 'target' in self.data_info['Variable_Type'].values:
            # For every target column
            for col in self.data_info[self.data_info['Variable_Type'].eq('target')]['Variable_Name']:
                # Drop column
                self.data.drop(col, axis=1, inplace=True)
                
        # 3. Save observed dosage
        col = self.data_info[self.data_info['Variable_Type'].eq('dosage')]['Variable_Name']
        # Save observed dosage
        self.dosage = self.data[col].to_numpy()
        # Drop dosage from dataframe
        self.data.drop(col, axis=1, inplace=True)
        
        # 4. Dummify categorical variables
        for col in self.data_info[self.data_info['Variable_Type'].eq('cat')]['Variable_Name']:
            # Dummy-fy column
            self.data = pd.concat([self.data, pd.get_dummies(self.data[col], prefix=col)], axis=1)
            # Drop original column
            self.data.drop(col, axis=1, inplace=True)
            
        # 5. Generate np data
        self.data = self.data.to_numpy()
        
        # 6. Normalize data
        self.data = self.normalize_data(self.data)
        
        ############################
        # Finalize data generation #
        ############################
        
        # Specify number of weights
        self.num_weights = 5
        
        # Initialize weight array 'v'
        self.v = np.zeros(shape=(self.num_weights,
                                 self.data.shape[1]))
        
        # Generate weights
        for i in range(self.num_weights):
            self.v[i] = np.random.uniform(0, 1, size=(self.data.shape[1]))
            self.v[i] = self.v[i] / np.linalg.norm(self.v[i])
            
        # Generate dataset
        self.dataset, self.dataset_train, self.dataset_val, self.dataset_test = self.generate_dataset(self.data)
        
    ####################
    # Helper functions #
    ####################
    
    # Normalization function
    def normalize_data(self, features):
        # Normalize based on min/max values per variable
        x = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0))

        # Divide by norm of variable vector
        for i in range(x.shape[0]):
            x[i] = x[i] / np.linalg.norm(x[i])

        return x
    
    # Generate dataset
    def generate_dataset(self, all_obs):
        # Set up as dict
        dataset = dict()
        dataset['x'] = []
        dataset['y'] = []
        dataset['d'] = []
        # Meta data
        dataset['gt'] = self.gt
        dataset['v'] = self.v
        
        # Iterate over observations to generate dosages
        for id, obs in tqdm(enumerate(all_obs), leave=False, desc="Generate dosages"):
            # Generate observation
            dosage = self.generate_dosage(x=obs, id=id)
            
            # Append to dict
            dataset['x'].append(obs)
            dataset['d'].append(dosage)
            
        # Normalize 'd' to [0,1]
        maxi = max(dataset['d'])
        mini = min(dataset['d'])
        dataset['d'] = [((x - mini)/(maxi - mini)) for x in dataset['d']]
        
        # Iterate over observations to generate outcomes
        for id, obs in tqdm(enumerate(all_obs), leave=False, desc="Generate outcomes"):
            # Generate observation
            outcome = get_outcome(obs, dataset['d'][id], self.v, self.gt)
            
            # Add noise
            outcome = np.clip(a=(outcome + np.random.normal(0, self.noise_std)), a_min=0, a_max=1)
            
            # Run Bernoulli trial
            outcome = float(np.random.binomial(1,outcome,1))
            
            # Append to dict
            dataset['y'].append(outcome)
        
        # Transform values to np array
        for key in ['x', 'y', 'd']:
            dataset[key] = np.array(dataset[key])
            
        # Get split idxs
        train_idx, val_idx, test_idx = get_splits(num_obs=dataset['x'].shape[0],
                                                  test_frac=self.test_fraction,
                                                  val_frac=self.val_fraction)
        
        # Get splitted datasets
        dataset_train = dict()
        dataset_val = dict()
        dataset_test = dict()
        
        # Iterate
        for key in ['x', 'y', 'd']:
            dataset_train[key] = dataset[key][train_idx]
            dataset_val[key] = dataset[key][val_idx]
            dataset_test[key] = dataset[key][test_idx]
            
        # Set metadata
        dataset_train['v'] = dataset['v']
        dataset_val['v'] = dataset['v']
        dataset_test['v'] = dataset['v']
        
        dataset_train['gt'] = dataset['gt']
        dataset_val['gt'] = dataset['gt']
        dataset_test['gt'] = dataset['gt']
        
        return dataset, dataset_train, dataset_val, dataset_test
            
    def get_true_dr(self, observation):
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        
        # Get the dr curve
        dr_curve = np.array([get_outcome(observation, d, self.v, self.gt) for d in treatment_strengths])
        
        return dr_curve
    
    def get_treatments(self):
        # Save treatment strengths
        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
        return treatment_strengths
            
    # Generate observation
    def generate_dosage(self, x, id):
        # Assign observed dosage if needed
        if self.confounding_bias == -1:
            dosage = self.dosage[id].item()
        else:
            # Calculate alpha
            alpha = 1 + self.confounding_bias
            # Calculate mode
            mode = (np.dot(x, self.v[4]))
            # Get beta
            beta = compute_beta(alpha, mode)
            # Get dosage
            dosage = np.random.beta(alpha, beta)
            
        return dosage
        
def get_splits(num_obs, val_frac, test_frac):
    # Get indexes
    indexes = [i for i in range(num_obs)]
    # Get number of observations per test/val
    num_test = int(num_obs * test_frac)
    num_val = int(num_obs * val_frac)
    
    # Get indexes
    rest_idx, test_idx = train_test_split(indexes, test_size=num_test)
    train_idx, val_idx = train_test_split(rest_idx, test_size=num_val)
    
    return train_idx, val_idx, test_idx

# Beta computation
# From Bica (2020)
def compute_beta(alpha, modal_rate):
    beta = (alpha - 1.0) / float(modal_rate) + (2.0 - alpha)
    return beta

# Get outcome
def get_outcome(x, d, v, gt):
    if gt == 'rc':
        y = generalized_logistic(dosage=d,
                                 input1=np.dot(x, v[0]),
                                 input2=np.dot(x, v[1]),
                                 input3=np.dot(x, v[2]),
                                 input4=np.dot(x, v[3]))
    else:    
        y = double_logistic(dosage=d,
                            input1=np.dot(x, v[0]),
                            input2=np.dot(x, v[1]),
                            input3=np.dot(x, v[2]),
                            input4=np.dot(x, v[3]))

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