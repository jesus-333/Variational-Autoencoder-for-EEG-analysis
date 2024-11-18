"""
Compare the SDTW with the block version of the SDTW.
As input signals use a trial of dataset 2a and its reconstruction with hvEEGNEt
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
import numpy as np

from library.training import soft_dtw_cuda, loss_function
from library.analysis import support

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj = 3
use_test_set = False

block_size = 125

trials_to_use = 10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Setup

# SDTW Loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtw_loss_function = soft_dtw_cuda.SoftDTW(use_cuda = torch.cuda.is_available(), gamma = 1)

# Load data
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

# Matrix to save the results (SDTW)
error_sdtw                      = np.zeros((trials_to_use, len(dataset.ch_list)))
error_sdtw_norm_by_time_samples = np.zeros((trials_to_use, len(dataset.ch_list)))
error_sdtw_norm_by_n_elements   = np.zeros((trials_to_use, len(dataset.ch_list)))

# Matrix to save the results (SDTW block)
error_sdtw_block = np.zeros((trials_to_use, len(dataset.ch_list)))
error_sdtw_block_norm_by_block_number = np.zeros((trials_to_use, len(dataset.ch_list)))
error_sdtw_block_norm_by_n_elements   = np.zeros((trials_to_use, len(dataset.ch_list)))

T = train_dataset.data.shape[-1]
if T % block_size == 0 : block_number = T / block_size 
else : raise ValueError("T is not divisible by block_size. T % block_size = {}".format(T % block_size))

# Load weights
path_weight = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep 3/model_80.pth'.format(subj)
# path_weight = 'Saved Model/test_SDTW_divergence/S{}/model_{}.pth'.format(subj,epoch) # TODO remember remove
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
model_hv.to(device)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Compute losses

#%% C
for i in range(trials_to_use) :
    print("i = {}".format(i))
    n_trial = np.random.randint(0, len(dataset))

    x, label = dataset[n_trial]
    x = x.unsqueeze(0).to(device)
    x_r = model_hv.reconstruct(x)

    for j in range(len(dataset.ch_list)) : # Iterate through EEG Channels
        print("\tj = {}".format(j))
        x_ch = x[:, :, j, :].swapaxes(1, 2)
        x_r_ch = x_r[:, :, j, :].swapaxes(1, 2)
        # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
        # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
        # With this operation I obtain element of shape [B x T x D]
        
        # Compute normal DTW
        tmp_recon_loss = dtw_loss_function(x_ch, x_r_ch)
        error_sdtw[i,j] = tmp_recon_loss
        error_sdtw_norm_by_time_samples[i,j] = tmp_recon_loss / T
        error_sdtw_norm_by_n_elements[i, j] = tmp_recon_loss / (T * T)

        # Compute block DTW
        tmp_recon_loss = loss_function.block_sdtw(x_ch, x_r_ch, dtw_loss_function, block_size, soft_DTW_type = 3, normalize_by_block_size = False)
        error_sdtw_block[i, j] = tmp_recon_loss
        error_sdtw_block_norm_by_block_number[i, j] = tmp_recon_loss / block_number
        error_sdtw_block_norm_by_n_elements[i, j] = tmp_recon_loss / (block_size * T)
