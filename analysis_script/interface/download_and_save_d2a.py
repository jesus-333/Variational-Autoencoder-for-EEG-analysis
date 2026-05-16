"""
Download the dataset 2a of the BCI Competition IV and save each trial as a npy file.
The label will be saved a single array.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import os

from library.analysis import support
from library.config import config_dataset as cd

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

subj_for_data = 2

path_save = './data_interface/'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def save_dataset(dataset, subj : int, type_dataset : str, path_save : str) :

    path_save_dataset = f"{path_save}/S{subj}/{type_dataset}/"
    os.makedirs(path_save_dataset, exist_ok = True)

    label_list = []
    for i in range(len(dataset)) :
        trial, label = train_dataset[i]

        label_list.append(label)

        path_save_trial = f"{path_save_dataset}/trial_{i}.npy"
        np.save(path_save_trial, trial.squeeze().numpy())

    label_array = np.array(label_list)
    path_save_label = f"{path_save_dataset}/labels.npy"
    np.save(path_save_label, label_array)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_config = cd.get_moabb_dataset_config([subj_for_data])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , _ = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

save_dataset(train_dataset, subj_for_data, 'train', path_save)
save_dataset(test_dataset, subj_for_data, 'test', path_save)
    
