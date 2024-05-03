"""
Compute the PCA on the latent space of the encoded data.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import os

import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from library.analysis import support
from library.dataset import preprocess as pp

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Subject to use for the weights of trained network.
# E.g. if subj_train = 3 the script load in hvEEGNet the weights obtained after the training with data from subject 3
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_train_list = [3]

# Training repetition and epoch of the weights
repetition = 1
epoch = 80

# The PCA will be computed on the data of this subjects
subj_for_PCA_computation = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_for_PCA_computation = [1]

# If True the PCA is computed between array on the z sampled from the latent space.
# Otherwise it will be computed on the mu tensor
pca_on_z_samples = True

pca_components = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_most_important_features(data, pca_components : int = None):
    # Compute PCA
    pca = PCA(n_components = pca_components)
    pca.fit(data)

    # Get the most important original components for each of the PCA components
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(pca_components)]

    return most_important

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PCA Computation

# PCA object from sklearn
pca = PCA(n_components = pca_components)

# Matrix to saved data
a = 1

for i in range(len(subj_train_list)) : # Training subjects iteration
    # Get model
    subj_train = subj_train_list[i]
    dataset_config_train = cd.get_moabb_dataset_config([subj_train])
    dataset_config_train['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    _, _, _, model_hv = support.get_dataset_and_model(dataset_config_train, 'hvEEGNet_shallow')
    print("Subj train {}".format(subj_train))
    
    # Load weights
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(80, subj_train, repetition, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

    # Set model to evaluation mode
    model_hv.eval()
    
    # Array to save all the data of all the subjects inside a single variable (per session).
    x_eeg_session_1 = None
    x_eeg_session_2 = None

    for j in range(len(subj_for_PCA_computation)) : # Data subjects iteration
        # Get dataset for the subject
        subj_train = subj_for_PCA_computation[i]
        dataset_config_other_subj = cd.get_moabb_dataset_config([subj_train])
        dataset_config_other_subj['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
        dataset_session_1, _, dataset_session_2 = pp.get_dataset_d2a(dataset_config_other_subj)

        # Get data for the two sessions
        if x_eeg_session_1 is None :
            x_eeg_session_1 = dataset_session_1[:][0]
            x_eeg_session_2 = dataset_session_2[:][0]
        else :
            x_eeg_session_1 = np.concatenate((x_eeg_session_1, dataset_session_1[:][0]))
            x_eeg_session_2 = np.concatenate((x_eeg_session_2, dataset_session_2[:][0]))

    # Get the sample or the array of mean
    if pca_on_z_samples :
        x_embedding_session_1 = model_hv.encode(torch.from_numpy(x_eeg_session_1))[0].flatten(1)
        x_embedding_session_2 = model_hv.encode(torch.from_numpy(x_eeg_session_2))[0].flatten(1)
    else :
        x_embedding_session_1 = model_hv.encode(torch.from_numpy(x_eeg_session_1))[1].flatten(1)
        x_embedding_session_2 = model_hv.encode(torch.from_numpy(x_eeg_session_2))[1].flatten(1)

    important_features_session_1 = compute_most_important_features(x_embedding_session_1, pca_components)
    important_features_session_2 = compute_most_important_features(x_embedding_session_2, pca_components)


