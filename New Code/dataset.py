#%% Imports

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import moabb.datasets as mb
import moabb.paradigms as mp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_train_data(config):
    # Get data and labels
    data, labels = get_D2a_data(config, 'train')
    
    # Create Pytorch dataset
    full_dataset = EEG_Dataset(data, labels, config['normalize_trials'])
    
    # Split in train and validation set
    train_dataset, validation_dataset = split_dataset(full_dataset, config['percentage_split'])
    
    return train_dataset, validation_dataset

def get_test_data(config):
    data, labels = get_D2a_data(config, 'test')
    
    # Create Pytorch dataset
    test_dataset = EEG_Dataset(data, labels, config['normalize_trials'])
    
    return test_dataset

def get_D2a_data(config, type_dataset):
    check_config(config)
    
    # Select the dataset
    dataset = mb.BNCI2014001()

    # Select the paradigm (i.e. the object to download the dataset)
    paradigm = mp.MotorImagery()

    # Get the data
    raw_data, raw_labels = get_moabb_data(dataset, paradigm, config, type_dataset)
    
    # Select channels
    data = raw_data[:, 0:22, :]

    # Convert labels
    labels = convert_label(raw_labels)

    return data, labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset(Dataset):

    def __init__(self, data, labels, normalize = False):
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

def check_config(config):
    # Check the frequency filter settings
    if 'filter_data' not in config: config['filter_data'] = False

    if config['filter_data'] == True:
        if 'fmin' not in config or 'fmax' not in config: raise ValueError('If you want to filter the data you must specify the lower (fmin) and upper (fmax) frequency bands  of the filter')
    
    # Check the resampling settings
    if 'resample_data' not in config: config['resample_data'] = False

    if config['resample_data']:
        if 'resample_freq' not in config: raise ValueError('You must specify the resampling frequency (resample_freq)')
        if config['resample_freq'] <= 0: raise ValueError('The resample_freq must be a positive value')
        
    if 'subject_by_subject_normalization' not in config: config['subject_by_subject_normalization'] = False
        
def get_moabb_data(dataset, paradigm, config, type_dataset):
    """
    Return the raw data from the moabb package of the specified dataset and paradigm
    N.b. dataset and paradigm must be object of the moabb library
    """

    if config['resample_data']: paradigm.resample = config['resample_freq']

    if config['filter_data']: 
        paradigm.fmin = config['fmin']
        paradigm.fmax = config['fmax']
    
    paradigm.n_classes = config['n_classes']

    # Get the raw data
    raw_data, raw_labels, info = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])
        
    # Select train/test data
    if type_dataset == 'train':
        idx_type = info['session'].to_numpy() == 'session_T'
    elif type_dataset == 'test':
        idx_type = info['session'].to_numpy() == 'session_E'

    raw_data = raw_data[idx_type]
    raw_labels = raw_labels[idx_type]

    return raw_data, raw_labels

def convert_label(raw_labels):
    """
    Convert the "raw" label obtained from the moabb dataset into a numerical vector where to each label is assigned a number
    """
    
    # Create a vector of the same lenght of the previous labels vector
    new_labels = np.zeros(len(raw_labels))
    
    # Create a list with all the possible labels
    labels_list = np.unique(raw_labels)
    
    # Iterate through the possible labels
    for i in range(len(labels_list)):
        # Get the label
        label = labels_list[i]
        idx_label = raw_labels == label
        
        # Assign numerical label
        new_labels[idx_label] = i

    return new_labels


def split_dataset(full_dataset, percentage_split):
    """
    Split a dataset in 2 
    """

    size_train = int(len(full_dataset) * percentage_split) 
    size_val = len(full_dataset) - size_train
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [size_train, size_val])
    
    return train_dataset, validation_dataset        

#%% End file
