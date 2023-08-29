"""
Computation of the reconstruction error for each trial
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.training.soft_dtw_cuda import SoftDTW

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

subj_list = [5]
loss = 'dtw'
epoch_list = [20, 40, 'BEST']
epoch_list = [40]

def get_dataset_and_model(subj_list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)
    device = 'cpu'

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Create model (hvEEGNet)
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model_config['use_classifier'] = False
    model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
    model_hv.to(device)

    return train_dataset, validation_dataset, test_dataset , model_hv

def compute_loss_dataset(dataset, model, path_weight):
    model.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)
    
    for i in range(len(dataset)):
        pass

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

