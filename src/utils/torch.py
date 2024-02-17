# Written by Christopher Bockel-Rickermann. Copyright (c) 2023

import torch

from torch.utils.data import Dataset

class torchDataset(Dataset):

    def __init__(self, dataset):

        # Save v vector
        self.v = dataset['v']
        # Assign values according to indices    
        self.x = torch.from_numpy(dataset['x']).type(torch.float32)
        self.y = torch.from_numpy(dataset['y']).type(torch.float32)
        self.d = torch.from_numpy(dataset['d']).type(torch.float32)
        # Save length
        self.length = dataset['x'].shape[0]
        # Save response type
        self.response = dataset['gt']
    
    # Define necessary fcts
    def get_data(self):
        return self.x, self.y, self.d
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.d[index]
    
    def __len__(self):
        return self.length