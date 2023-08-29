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

use_test_set = True
use_cuda = True if device == 'cuda' else False

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

def compute_loss_dataset(dataset, model, use_cuda):
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    recon_loss_matrix = np.zeros((len(dataset), dataset.ch_list))
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # Get the original signal and reconstruct it
            x = dataset[i].unsqueeze(0) # Add batch dimension
            x_r = model.reconstruct(x)
            
            # Compute the DTW channel by channels
            tmp_recon_loss = []
            for i in range(x.shape[2]): # Iterate through EEG Channels
                x_ch = x[:, :, i, :].swapaxes(1,2)
                x_r_ch = x_r[:, :, i, :].swapaxes(1,2)
                # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
                # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
                
                tmp_recon_loss.append(float(recon_loss_function(x_ch, x_r_ch)))

            recon_loss_matrix[i, :] = tmp_recon_loss

    return recon_loss_matrix

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

recon_loss_results = dict()

for subj in subj_list:
    recon_loss_results[subj] = dict()

    for epoch in epoch_list:
        train_dataset, validation_dataset, test_dataset , model_hv = get_dataset_and_model([subj])

        if use_test_set: dataset = test_dataset
        else: dataset = train_dataset
        
        # Load model weight
        path_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, subj, epoch)
        model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

        recon_loss_results[subj][epoch] = compute_loss_dataset(dataset, model_hv, path_weight)
