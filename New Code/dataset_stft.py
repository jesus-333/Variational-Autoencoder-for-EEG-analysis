"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset with EEG data trasformed through stft (or similar transformation like wavelet)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
from torch.utils.data import Dataset

"""
%load_ext autoreload
%autoreload 2

import dataset_stft as ds_stft
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset_stft(Dataset):

    def __init__(self, data, labels, ch_list, t, f):
        """
        data = data used for the dataset. Must have shape [Trials x channels x frequency samples x time samples]
        labels = labels of each trial
        ch_list = list with the name of the eeg channel
        t = time array with the value of the time instants after the stft
        f = frequency array with the value of the frequency bins after the stft    
        """

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = ch_list
        
        self.t = t
        self.f = f
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)
    
    def visualize_trial(self, idx_trial, ch):
        # TODO
        pass
    
#%% Check data

def main():
    import config_dataset as cd
    import preprocess as pp
    
    dataset_config = cd.get_moabb_dataset_config([3])
    dataset_config['stft_parameters'] = cd.get_config_stft()
    
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
    
    print(train_dataset[0][0].shape)
    print(train_dataset.ch_list)


if __name__ == "__main__":
    main()

#%% End file
