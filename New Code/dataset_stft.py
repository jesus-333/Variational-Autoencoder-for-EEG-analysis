"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset with EEG data trasformed through stft (or similar transformation like wavelet)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import preprocess as pp
import download
import support_function as sf

"""
%load_ext autoreload
%autoreload 2

import dataset as ds
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_train_data_d2a(config : dict):
    # Get data and labels
    data, labels, ch_list = download.get_D2a_data(config, 'train')
    data, t, f = pp.compute_stft(data, config)

    # Create Pytorch dataset
    full_dataset = EEG_Dataset_stft(data, labels)
    
    # Split in train and validation set
    train_dataset, validation_dataset = sf.split_dataset(full_dataset, config['percentage_split'])
    
    return train_dataset, validation_dataset

def get_test_data_d2a(config : dict):
    data, labels = download.get_D2a_data(config, 'test')
    data, t, f = pp.compute_stft(data, config)

    # Create Pytorch dataset
    test_dataset = EEG_Dataset_stft(data, labels)
    
    return test_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset_stft(Dataset):

    def __init__(self, data, labels):
        """
        data = data used for the dataset. Must have shape [Trials x channels x frequency samples x time samples]
        """

        # Transform data in torch array
        self.data = torch.from_numpy(data).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)

#%% End file
