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

import dataset_stft as ds_stft
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_train_data_d2a(config : dict):
    # Get data and labels
    data, labels, ch_list = download.get_D2a_data(config, 'train')
    data, t, f = pp.compute_stft(data, config)

    # By default the labels obtained through the moabb have value between 1 and 4. 
    # But Pytorch for 4 classes want values between 0 and 3
    labels -= 1

    # Create Pytorch dataset
    full_dataset = EEG_Dataset_stft(data, labels)
    
    # Split in train and validation set
    train_dataset, validation_dataset = sf.split_dataset(full_dataset, config['percentage_split_train_validation'])
    
    return train_dataset, validation_dataset

def get_test_data_d2a(config : dict):
    data, labels = download.get_D2a_data(config, 'test')
    data, t, f = pp.compute_stft(data, config)

    # By default the labels obtained through the moabb have value between 1 and 4. 
    # But Pytorch for 4 classes want values between 0 and 3
    labels -= 1

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
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)

#%% End file
