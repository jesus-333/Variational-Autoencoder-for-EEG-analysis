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

from library.analysis import support
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

# If True the distance is computed between array of sampled data (i.e. obtained sampling from the distribution)
# Otherwise the array of means will be used as representative of the data
sample_from_latent_space = True

# If True the distance is computed separately for each class
# TODO implement in a new version of the script
# divide_by_class = False

# If True compute again the distance even if it was already computed in past
force_computation = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load the precomputed matrix (if exist) otherwise created an empty matrix

path_save = 'Saved Results/wasserstein/raw_data/'

# Path to save/load the computed distances
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
    x_train, _ = train_dataset[:]
    u_samples = x_train[:].squeeze().flatten(1)

    for j in range(len(subj_for_distance_computation_list)) :
        # Get subj number
        subj_test = subj_for_distance_computation_list[j]

        if force_computation or distance_matrix_1[subj_train - 1, subj_test - 1] < -1 or distance_matrix_2[subj_train - 1, subj_test - 1] < -1 : # If I haven't computed the distance for this pair train data-other subj data then I computed it
            # Get the data to used for distance computation
            dataset_config_other_subj = cd.get_moabb_dataset_config([subj_test])
            dataset_config_other_subj['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
            dataset_session_1, _, dataset_session_2 = pp.get_dataset_d2a(dataset_config_other_subj)
            print("Subj test {}".format(subj_test))
            
            # Get data for the two sessions
            x_test_session_1, _ = dataset_session_1[:]
            x_test_session_2, _ = dataset_session_2[:]
            
            v_samples_1 = x_test_session_1[:].squeeze().flatten(1)
            v_samples_2 = x_test_session_2[:].squeeze().flatten(1)
        
            # Compute distance between distribution
            distance_train_session_1 = wasserstein_distance_nd(u_samples, v_samples_1)
            distance_train_session_2 = wasserstein_distance_nd(u_samples, v_samples_2)
        
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
