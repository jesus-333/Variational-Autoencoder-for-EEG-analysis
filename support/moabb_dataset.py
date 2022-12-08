#%% Imports

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import moabb.datasets as mb
import moabb.paradigms as mp

from config_file import split_dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_train_data(config):
    # Get data and labels
    data, labels = get_D2a_data(config)
    
    # Create Pytorch dataset
    full_dataset = EEG_Dataset(data, labels, config['normalize_trials'])
    
    # Split in train and validation set
    train_dataset, validation_dataset = split_dataset(full_dataset, config['percentage_split'])
    
    return train_dataset, validation_dataset

def get_D2a_data(config):
    check_config(config)
    
    # Select the dataset
    dataset = mb.BNCI2014001()

    # Select the paradigm (i.e. the object to download the dataset)
    paradigm = mp.MotorImagery()

    # Get the data
    raw_data, raw_labels = get_moabb_data(dataset, paradigm, config, 'train')
    
    # Select channels
    data = raw_data[:, 0:22, :]

    # Convert labels
    labels =  convert_label(raw_labels)

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
            self.data = (self.data - self.data.min())/(self.data.max() - self.data.min())

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other

def check_config(config):
    # Check the frequency filter settings
    if 'filter_data' not in config: config['filter_data'] = False

    if config['filter_data'] == True:
        if 'fmin' not in config or 'fmax' not in config: raise ValueError('If you want to filter the data you must specify the lower (fmin) and upper (fmax) frequency bands  of the filter')
    
    # Check the resampling settings
    if 'resample' not in config: config['resample'] = False

    if config['resample']:
        if 'resample_freq' not in config: raise ValueError('You must specify the resampling frequency (resample_freq)')
        if config['resample_freq'] <= 0: raise ValueError('The resample_freq must be a positive value')

def get_moabb_data(dataset, paradigm, config, type_dataset):
    """
    Return the raw data from the moabb package of the specified dataset and paradigm
    N.b. dataset and paradigm must be object of the moabb library
    """

    if config['resample']: paradigm.resample = config['resampe_freq']

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
        
        
