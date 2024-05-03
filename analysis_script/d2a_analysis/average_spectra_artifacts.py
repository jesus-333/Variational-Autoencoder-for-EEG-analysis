"""
Compute the average spectra of artifacts and NON-artifacts
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [4]
ch = 'C3'

use_test_set = True

nperseg = 500

plot_config = dict(
    figsize = (10, 8),
    fontsize = 24, 
    capsize = 3,
    alpha = 0.25,
    color = 'black',
    save_fig = True
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
plt.rcParams.update({'font.size': plot_config['fontsize']})

artifacts_map_train_original = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_train.csv').T.to_numpy()[1:, :]
artifacts_map_test_original  = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_test.csv').T.to_numpy()[1:, :]

subj_to_idx = np.arange(9)

if use_test_set: artifacts_map = artifacts_map_test_original
else : artifacts_map = artifacts_map_train_original 

for i in range(len(subj_list)):
    subj = subj_list[i]
    subj_idx = subj_to_idx[subj]

    idx_artifacts = artifacts_map[subj_idx] == 1
    idx_NON_artifacts = artifacts_map[subj_idx] == 0

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    if use_test_set:
        data = test_dataset.data.squeeze()
        dataset_string = 'test'
    else:
        data = train_dataset.data.squeeze()
        dataset_string = 'train'

    data_artifacts = data[idx_artifacts]
    data_NON_artifacts = data[idx_NON_artifacts]

    idx_ch = train_dataset.ch_list == ch

    average_spectra_artifacts, std_spectra_artifacts = support.compute_average_spectra(data_artifacts, nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    average_spectra_NON_artifacts, std_spectra_NON_artifacts = support.compute_average_spectra(data_NON_artifacts, nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    

