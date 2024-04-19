"""
This script produce a figure similar to Fig. 5 of Barandas et al. 2024 (https://www.sciencedirect.com/science/article/pii/S1566253523002944)

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import numpy as np
import matplotlib.pyplot as plt

from library.analysis import dtw_analysis, support
from library.dataset import preprocess as pp

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Subject to use for the weights of trained network.
# E.g. if subj_train = 3 the script load in hvEEGNet the weights obtained after the traiing with data from subject 3
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_train_list = [3]

# Training repetition and epoch of the weights
repetition = 5
epoch = 80

# The wesserstein distance will be computed respect this subjects
subj_for_distance_computation_list  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_for_distance_computation_list  = [1]

# If True the distance is computed between array of sampled data (i.e. obtained sampling from the distribution)
# Otherwise the array of means will be used as representative of the data
sample_from_latent_space = False

# If True the distance is computed separately for each class
# TODO implement in a new version of the script
# divide_by_class = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(subj_train_list)) :
    # Get model
    subj_train = subj_train_list[i]
    dataset_config_train = cd.get_moabb_dataset_config([subj_train])
    dataset_config_train['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, _, model_hv = support.get_dataset_and_model(dataset_config_train, 'hvEEGNet_shallow')

    # Load weights

    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(80, subj_train, repetition, epoch)
    
    # Get tensor with all training data
    x_train, _ = train_dataset[:]
    
    # Compute latent space representation
    # _, mu_list_train_data, log_var_list_train_data, _, _ = model_hv(x_train)
    # 
    # # Get representation of the deeper latent space
    # mu_train_data = mu_list_train_data[0]
    # log_var_train_data = log_var_train_data[0]
    
    # (OPTIONAL) Sample from the latent space
    a = model_hv.encode(x_train, return_distribution = sample_from_latent_space)

    if sample_from_latent_space :
        pass
    else : 
        pass


    for i in range(len(subj_for_distance_computation_list)) :
        # Get the data to used for distance computation
        subj_test = subj_for_distance_computation_list[i]
        dataset_config_other_subj = cd.get_moabb_dataset_config([subj_train])
        dataset_config_other_subj['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
        dataset_session_1, _, dataset_session_2 = pp.get_dataset_d2a(dataset_config_other_subj)

        #

