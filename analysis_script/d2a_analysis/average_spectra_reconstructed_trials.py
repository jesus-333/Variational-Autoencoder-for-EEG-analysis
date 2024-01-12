"""
Plot the average spectra for the artifacts and NON artifacts for dataset 2a.
artifacts and non artifacts are selected based on the the expert classification
"""
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import torch

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
repetition = 17
epoch = 80

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [4]
ch = 'C3'

use_test_set = True

nperseg = 500

plot_config = dict(
    figsize = (14, 8),
    fontsize = 20, 
    capsize = 3,
    alpha = 0.25,
    color = 'black',
    add_title = True,
    use_different_figure = False,
    save_fig = True
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})

# Get position of the artifacts
artifacts_map_train = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_train.csv').T.to_numpy()[1:, :]
artifacts_map_test = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_test.csv').T.to_numpy()[1:, :]

for subj in subj_list:

    subj_idx = subj - 1

    # Get the data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')


    subject_artifacts_map_train = artifacts_map_train[subj_idx]
    subject_artifacts_map_test = artifacts_map_test[subj_idx]

    x_train, label = train_dataset[:]
    x_test, label = test_dataset[:]
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

    # (OPTIONAL) Use CUDA
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        x_test = x_test.cuda()
        model_hv.cuda()
    
    # Reconstruct the data
    x_r_train = model_hv.reconstruct(x_train.unsqueeze(0)).squeeze()
    x_r_test = model_hv.reconstruct(x_test.unsqueeze(0)).squeeze()

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get average spectra for artifacts/NON artifacts

    idx_ch = train_dataset.ch_list == ch
    
    average_spectra_train, std_spectra_train, f = support.compute_average_spectra(x_r_train, nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    average_spectra_test, std_spectra_test, f = support.compute_average_spectra(x_r_test, nperseg = nperseg, fs = 250, idx_ch = idx_ch)

    average_spectra_list = [average_spectra_train, average_spectra_test]
    std_spectra_list = [std_spectra_train, std_spectra_test]
    label_list = ["Train", "Test"]

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    # Create figures
    if plot_config['use_different_figure']:
        fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize'])
    else:
        fig_freq, ax_freq_list = plt.subplots(1, 2, figsize = plot_config['figsize'])
        idx_figures = [0, 1] 

    for i in range(4):
        average_spectra = average_spectra_list[i]
        std_spectra = std_spectra_list[i]

        ax_freq = ax_freq_list[idx_figures[i]]


        ax_freq.plot(f, average_spectra,
                     color = plot_config['color'], label = label_list[i]
                     )
        ax_freq.fill_between(f, average_spectra + std_spectra, average_spectra - std_spectra, 
                             color = plot_config['color'], alpha = plot_config['alpha']
                             )
        ax_freq.set_xlabel("Frequency [Hz]")
        ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$] (S{})".format(subj))
        # ax_freq.legend()
        ax_freq.grid(True) 
        ax_freq.set_xlim([0, 80])
        ax_freq.set_ylim([0, 30])
        if plot_config['add_title']: ax_freq.set_title(label_list[i])
            
        ax_freq.set_ylim(bottom = -1)

        fig_freq.tight_layout()
        fig_freq.show()

