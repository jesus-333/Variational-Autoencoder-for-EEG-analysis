"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset with EEG data in format channels x time samples
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch.utils.data import Dataset

"""
%load_ext autoreload
%autoreload 2

import dataset as ds
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset(Dataset):

    def __init__(self, data, labels, ch_list, normalize = -1):
        """
        data = data used for the dataset. Must have shape [Trials x 1 x channels x time samples]
        Note that if you use normale EEG data depth dimension (the second axis) has value 1. 
        """
        # Transform data in torch array
        self.data = torch.from_numpy(data).unsqueeze(1).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = ch_list
        
        # (OPTIONAL) Normalize
        if normalize == 1:
            self.minmax_normalize_all_dataset(-1, 1)
        elif normalize == 2:
            self.normalize_channel_by_channel(-1, 1)
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)

    def minmax_normalize_all_dataset(self, a, b):
        """
        Normalize the entire dataset between a and b.
        """
        self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min())) * (b - a) + a 


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

class EEG_Dataset_ChWi(Dataset):

    def __init__(self, data, labels = None, ch_list : list = None):
        """
        Dataset used for the ChWi autoencoder. All EEG signals are saved independently from the trial.
        This means that an EEG matrix of shape B x 1 x C x T will be saved as a matrix of shape (B * C) x T

        @param data: data used for the dataset. Must have shape [Trials x 1 x C x T] with C = channels and T = time samples
        """

        # Get data
        self.data = torch.from_numpy(data).unsqueeze(1).float()

        # (OPTIONAL) Saved labels and channels
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.ch_list = ch_list
        
            
    def __getitem__(self, idx : int):
        if self.labels is not None :
            return self.data[idx], self.labels[idx]
        else :
            return self.data[idx], None
    
    def __len__(self):
        return len(self.labels)

#%% End file
