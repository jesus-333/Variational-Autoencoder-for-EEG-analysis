"""
Compute the wasserstein distance between the distribution encoded by the latent space

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import os

from scipy.stats import wasserstein_distance_nd
import numpy as np
from numba import prange

from library.analysis import support, normalize
from library.dataset import preprocess as pp

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Subject to use for the weights of trained network.
# E.g. if subj_train = 3 the script load in hvEEGNet the weights obtained after the traiing with data from subject 3
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_train_list = [3, 9]

# The wesserstein distance will be computed respect this subjects
subj_for_distance_computation_list  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_for_distance_computation_list  = [1]

# If True compute again the distance even if it was already computed in past
force_computation = True

normalization_type = -1
# 1 : z-score all subject data (using only training data)
# 2 : z-score trial per trial
# 3 : z-score trial per trial and channel per channel
# 4 : z-score channel per channel considering all the trials (using only training data). This means that to normalize a channel I take for all 288 repetitions the specific channel.
# All other values : No normalization

concatenate_channels = True # If true concatenate all channels of all trials in a single matrix. In this way I obtain a matrix of shape (288 * 22) x 1000, instead of a matrix of shape 288 x (22 * 1000)
 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load the precomputed matrix (if exist) otherwise created an empty matrix

path_save = 'Saved Results/wasserstein/raw_data/'

if normalization_type == 1 : path_save += 'z_score/'
elif normalization_type == 2 : path_save += 'z_score_trial_by_trial/'
elif normalization_type == 3 : path_save += 'z_score_ch_by_ch/'
elif normalization_type == 4 : path_save += 'z_score_trial_by_trial_ch_by_ch/'
else : path_save += 'no_normalization/'

# Path to save/load the computed distances
if concatenate_channels :
    path_save_session_1 = path_save + 'distance_session_1_to_session_1_raw_data_concatenate_channels.npy'
    path_save_session_2 = path_save + 'distance_session_1_to_session_2_raw_data_concatenate_channels.npy'
else :
    path_save_session_1 = path_save + 'distance_session_1_to_session_1_raw_data.npy'
    path_save_session_2 = path_save + 'distance_session_1_to_session_2_raw_data.npy'

# Check if the file exist otherwise creates it
if os.path.isfile(path_save_session_1) :
    distance_matrix_1 = np.load(path_save_session_1)
else :
    distance_matrix_1 = np.ones((9, 9)) * -10

if os.path.isfile(path_save_session_2) :
    distance_matrix_2 = np.load(path_save_session_2)
else :
    distance_matrix_2 = np.ones((9, 9)) * -10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute wasserstein distance between train data and other subjects data

for i in range(len(subj_train_list)) :
    # Get model
    subj_train = subj_train_list[i]
    dataset_config_train = cd.get_moabb_dataset_config([subj_train])
    dataset_config_train['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, _, _ = support.get_dataset_and_model(dataset_config_train, 'hvEEGNet_shallow')
    print("Subj train {}".format(subj_train))
    
    # Get tensor with all training data
    x_train  = train_dataset[:][0].squeeze()
    
    # (OPTIONAL) Normalize
    if normalization_type == 1 :
        x_train = normalize.z_score(x_train.numpy())
    elif normalization_type == 2 :
        x_train = normalize.z_score_trial_by_trial(x_train.numpy())
    elif normalization_type == 3 :
        x_train = normalize.z_score_trial_by_trial_ch_by_ch(x_train.numpy())
    elif normalization_type == 4 :
        x_train = normalize.z_score_ch_by_ch(x_train.numpy())
    
    # Get a 2D distance_matrix
    if concatenate_channels : # (B * C) x T
        u_samples = x_train[:].squeeze().reshape(x_train.shape[0] * x_train.shape[1], -1)
    else : # B x (C * T)
       # Flatten extra dimension to obtain a 2D matrix
        u_samples = x_train[:].squeeze().reshape(x_train.shape[0], -1)
        # N.b. I use reshape instead of flatten beacuse the behaviour is consistent with both numpy and tensorflow

    for j in prange(len(subj_for_distance_computation_list)) :
        # Get subj number
        subj_test = subj_for_distance_computation_list[j]

        if force_computation or distance_matrix_1[subj_train - 1, subj_test - 1] < -1 or distance_matrix_2[subj_train - 1, subj_test - 1] < -1 : # If I haven't computed the distance for this pair train data-other subj data then I computed it
            # Get the data to used for distance computation
            dataset_config_other_subj = cd.get_moabb_dataset_config([subj_test])
            dataset_config_other_subj['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
            dataset_session_1, _, dataset_session_2 = pp.get_dataset_d2a(dataset_config_other_subj)
            print("Subj test {}".format(subj_test))
            
            # Get data for the two sessions
            x_test_session_1 = dataset_session_1[:][0].squeeze()
            x_test_session_2 = dataset_session_2[:][0].squeeze()

            # (OPTIONAL) Normalize
            if normalization_type == 1 :
                x_mean_train, x_std_train = float(np.mean(x_train)), np.std(x_train)
                x_test_session_1 = normalize.z_score(x_test_session_1.numpy(), x_mean_train, x_std_train)
                x_test_session_2 = normalize.z_score(x_test_session_2.numpy(), x_mean_train, x_std_train)
            elif normalization_type == 2 :
                x_test_session_1 = normalize.z_score_trial_by_trial(x_test_session_1.numpy())
                x_test_session_2 = normalize.z_score_trial_by_trial(x_test_session_2.numpy())
            elif normalization_type == 3 :
                x_test_session_1 = normalize.z_score_trial_by_trial_ch_by_ch(x_test_session_1.numpy())
                x_test_session_2 = normalize.z_score_trial_by_trial_ch_by_ch(x_test_session_2.numpy())
            elif normalization_type == 4 :
                x_mean_train, x_std_train = float(np.mean(x_train)), np.std(x_train)
                x_test_session_1 = normalize.z_score_ch_by_ch(x_test_session_1.numpy(), x_mean_train, x_std_train)
                x_test_session_2 = normalize.z_score_ch_by_ch(x_test_session_2.numpy(), x_mean_train, x_std_train)
            
            # Get a 2D matrix
            if concatenate_channels : # (B * C) x T
                v_samples_1 = x_test_session_1[:].squeeze().reshape(x_test_session_1.shape[0] * x_test_session_1.shape[1], -1)
                v_samples_2 = x_test_session_2[:].squeeze().reshape(x_test_session_2.shape[0] * x_test_session_2.shape[1], -1)
            else : # B x (C * T)
                v_samples_1 = x_test_session_1[:].squeeze().reshape(x_test_session_1.shape[0], -1)
                v_samples_2 = x_test_session_2[:].squeeze().reshape(x_test_session_2.shape[0], -1)
        
            # Compute distance between distribution
            distance_train_session_1 = wasserstein_distance_nd(np.asarray(u_samples), np.asarray(v_samples_1))
            distance_train_session_2 = wasserstein_distance_nd(np.asarray(u_samples), np.asarray(v_samples_2))
        
            # Saved values
            distance_matrix_1[subj_train - 1, subj_test - 1] = distance_train_session_1
            distance_matrix_2[subj_train - 1, subj_test - 1] = distance_train_session_2
            
            # Folder to save results
            os.makedirs(path_save, exist_ok = True)
            
            # Save the data
            np.save(path_save_session_1, distance_matrix_1)
            np.save(path_save_session_2, distance_matrix_2)
        else : # If I already computed the distance skip this iteration and directly load the precomputed data
            continue
