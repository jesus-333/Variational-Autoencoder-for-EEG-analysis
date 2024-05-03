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
from torch.utils.data import DataLoader

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.training.soft_dtw_cuda import SoftDTW

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

subj_list = [1,2,3,4,5,6,7,8,9]
loss = 'dtw'
# epoch_list = [20, 40, 'BEST']
epoch_list = [40]

use_test_set = False

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def compute_loss_dataset(dataset, model, device, batch_size = 32):
    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    recon_loss_matrix = np.zeros((len(dataset), len(dataset.ch_list)))
    
    with torch.no_grad():
        model.to(device)
        dataloader = DataLoader(dataset, batch_size = batch_size)
        i = 0
        for x_batch, _ in dataloader:
            # Get the original signal and reconstruct it
            x = x_batch.to(device)
            x_r = model.reconstruct(x)
            
            # Compute the DTW channel by channels
            tmp_recon_loss = np.zeros((x_batch.shape[0], len(dataset.ch_list)))
            for j in range(x.shape[2]): # Iterate through EEG Channels
                x_ch = x[:, :, j, :].swapaxes(1,2)
                x_r_ch = x_r[:, :, j, :].swapaxes(1,2)
                # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
                # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
                
                tmp_recon_loss[:, j] = recon_loss_function(x_ch, x_r_ch).cpu()

            recon_loss_matrix[(i * batch_size):((i * batch_size) + x.shape[0]), :] = tmp_recon_loss
            i += 1

    return recon_loss_matrix

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

recon_loss_results = dict()

for subj in subj_list:
    print(subj)
    recon_loss_results[subj] = dict()
    train_dataset, validation_dataset, test_dataset , model_hv = get_dataset_and_model([subj])
    for epoch in epoch_list:
        if use_test_set: dataset = test_dataset
        else: dataset = train_dataset
        
        # Load model weight
        path_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, subj, epoch)
        model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

        recon_loss_results[subj][epoch] = compute_loss_dataset(dataset, model_hv, device, batch_size) / 1000
