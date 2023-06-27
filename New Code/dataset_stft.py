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
#%% PyTorch Dataset

class EEG_Dataset_stft(Dataset):

    def __init__(self, data, labels, ch_list):
        """
        data = data used for the dataset. Must have shape [Trials x channels x frequency samples x time samples]
        """

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = ch_list
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)
    
    def visualize_trial(self, idx_trial, ch):
        pass

#%% End file
