"""
Compute the average DTW for between the reconstruction obtained with only the deepest latent space (z1)
Can work both intra-subject or cross subject
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
import itertools

from library.analysis import support
from library.config import config_dataset as cd 
from library.training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_data = 9
subj_weights = 2 # For intra-subject set subj_data == subj_weights

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
dataset_config = cd.get_moabb_dataset_config([subj_data])
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
intra_subj_DTW_values = np.zeros((len(ch_list), len(dataset), len(dataset))) # Save DTW

# Load weights
path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj_weights, repetition, epoch)
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
#%% DTW Computation

# DTW loss funcion
recon_loss_function = SoftDTW(use_cuda = True if torch.cuda.is_available() else False, normalize = False)

for j in range(len(list_of_combinations)): 
    print("Iterations : {}%".format(round((j + 1) / len(list_of_combinations) * 100, 3)))

    idx_0 = list_of_combinations[j][0]
    idx_1 = list_of_combinations[j][1]

    x_0 = x_r_dataset[idx_0].unsqueeze(0).unsqueeze(0)
    x_1 = x_r_dataset[idx_1].unsqueeze(0).unsqueeze(0)

    # Compute the DTW channel by channels
    for k in range(x_0.shape[2]): # Iterate through EEG Channels
        x_0_ch = x_0[:, :, k, :].swapaxes(1,2)
        x_1_ch = x_1[:, :, k, :].swapaxes(1,2)
        # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
        # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
        
        tmp_recon_loss = float(recon_loss_function(x_0_ch, x_1_ch).cpu()) / x_0.shape[-1]

        intra_subj_DTW_values[k, idx_0, idx_1] = tmp_recon_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Save results
path_save = "Saved Results/intra_subjects_study/"

if subj_weights == subj_weights: # Intra subject
    path_save += "intra_subject_dtw_S{}.npy".format(subj_weights)
else:
    path_save += "cross_subject_dtw_source_S{}_target_s{}.npy".format(subj_weights, subj_data)
    
np.save(path_save, intra_subj_DTW_values)