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

        if len(data.shape) != 4 or data.shape[1] != 1 :
            raise ValueError("The input shape of data must be [Trials x 1 x channels x time samples]. Current shape {}".format(data.shape))

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = ch_list
        
        # (OPTIONAL) Normalize
        if normalize == 1:
            self.minmax_normalize_all_dataset(-1, 1)
        elif normalize == 2:
            self.normalize_channel_by_channel(-1, 1)
            
    def __getitem__(self, idx : int) -> tuple[torch.tensor, torch.tensor] :
        return self.data[idx], self.labels[idx]
    
    def __len__(self) -> int :
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
        This means that an EEG matrix of shape B x C x T will be saved as a matrix of shape (B * C) x 1 x T
        The dimension of size 1 is used for the depth map of 1d convolution.

        @param data: data used for the dataset. Must have shape [Trials x C x T] with C = channels and T = time samples
        """

        # Get data
        if 'torch' in str(type(data)) :
            self.data = data.float()
        else :
            self.data = torch.from_numpy(data).float()

        # Reshape matrix
        self.data = self.data.reshape(-1, 1, self.data.shape[2])

        # Saved labels
        if labels is None :
            # If no labels is passed an array of ones is created. This is only used to keep dataset consistent with training functions
            self.labels = torch.ones(self.data.shape[0])
        else :
            if 'torch' in str(type(data)) :
                self.labels = labels.long()
            else :
                self.labels = torch.from_numpy(labels).long()

        # Saved channels
        self.ch_list = ch_list

        # Used to return only data. False by default. It can be set to True after dataset creation in the code.
        self.return_only_data = False
        
    def __getitem__(self, idx : int):
        if self.return_only_data :
            return self.data[idx]
        else :
            return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return self.data.shape[0]

#%% End file
