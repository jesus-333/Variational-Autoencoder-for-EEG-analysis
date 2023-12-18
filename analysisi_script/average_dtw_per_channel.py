"""
Compute the average DTW for a specific subject between the reconstruction obtained with only the deepest latent space (z1)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
import itertools

from library.analysis import support
from library.config import config_dataset as cd 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj = 2

ch_list = np.asarray(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz','P2', 'POz'])
ch_list = np.asarray(['Fz', 'C2', 'P1'])

use_test_dataset = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 72

# Parameter for the weights to load
tot_epoch_training = 80 # Total number of training epoch (e.g. 80 used the weights of the networks obtained with the training run of 80 epochs)
repetition = 1 # Select training repetition
epoch = 80 # Select the epoch of the weights. Must be less or equal than tot_epoch_training

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Create dataset and model

# Get the datasets and model
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Select dataset to use
dataset = test_dataset if use_test_dataset else train_dataset

# Get the indices of the channels where DTW will be computed
ch_list_dataset = dataset.ch_list
idx_ch = np.zeros(len(ch_list_dataset)) != 0
for ch in ch_list: idx_ch = np.logical_or(idx_ch, ch_list_dataset == ch)

# Tensor to save the data
x_r_dataset = torch.asarray([]).to(device) # Save reconstructed data
tensor_DTW = torch.zeros((len(ch_list), len(dataset), len(dataset))) # Save DTW

# Load weights
path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

# Dataloader
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

# Move model to device (cpu/cuda)
model_hv.to(device)

# List of the combinations inside the batch 
list_of_combinations = list(set(itertools.combinations(np.arange(len(dataset)), 2)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Signals reconstruction

for sample_data_batch, sample_label_batch in dataset_loader:
    # Reconstruct the signal using only z1
    x = sample_data_batch.to(device)
    _, _, x_z1 = support.compute_latent_space_different_resolution(model_hv, x)

    # Select the channels where DTW will be computed (P.s. with the method above the x_z1 has shape B x C x T)
    x_z1 = x_z1[:, idx_ch, :]
    
    x_r_dataset = torch.cat((x_r_dataset, x_z1))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% 



    


