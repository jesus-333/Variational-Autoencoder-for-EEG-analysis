"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import preprocess as pp
import download

"""
%load_ext autoreload
%autoreload 2

import dataset as ds
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_train_data_d2a(config):
    # Get data and labels
    data, labels, ch_list = download.get_D2a_data(config, 'train')
    
    # Create Pytorch dataset
    full_dataset = EEG_Dataset(data, labels, config['normalize_trials'])
    
    # Split in train and validation set
    train_dataset, validation_dataset = split_dataset(full_dataset, config['percentage_split'])
    
    return train_dataset, validation_dataset

def get_test_data_d2a(config):
    data, labels = download.get_D2a_data(config, 'test')
    
    # Create Pytorch dataset
    test_dataset = EEG_Dataset(data, labels, config['normalize_trials'])
    
    return test_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset(Dataset):

    def __init__(self, data, labels, normalize = False):
        """
        data = data used for the dataset. Must have shape [Trials x 1 x channels x time samples]
        Note that if you use normale EEG data depth dimension (the second axis) has value 1. 
        """
        # Transform data in torch array
        self.data = torch.from_numpy(data).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()
        
        # (OPTIONAL) Normalize
        if normalize:
            self.normalize_channel_by_channel(-1, 1)
            
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)
    
    def normalize_channel_by_channel(self, a, b):
        """
        Normalize each channel so the value are between a and b
        """
        
        # N.b. self.data.shape = [trials, 1 , channels, eeg samples]
        # The dimension with the 1 is needed because the conv2d require a 3d tensor input
        
        for i in range(self.data.shape[0]): # Cycle over samples
            for j in range(self.data.shape[2]): # Cycle over channels
                tmp_ch = self.data[i, 0, j]
                
                normalize_ch = ((tmp_ch - tmp_ch.min()) / (tmp_ch.max() - tmp_ch.min())) * (b - a) + a
                
                self.data[i, 0, j] = normalize_ch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other


#%% End file
