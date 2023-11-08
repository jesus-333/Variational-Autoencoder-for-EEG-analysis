"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Visualize multiple trial in a grid 3 x 3
The trial 
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [5]
ch_list = ['FC3', 'Fz', 'FC4', 'C5', 'Cz', 'C6', 'P1', 'POz', 'P2']
ch_list_for_grid = [['FC3', 'Fz', 'FC4'], ['C5', 'Cz', 'C6'], ['P1', 'POz', 'P2']]

use_test_set = False

n_trial_to_plot = 3

nperseg = 500

plot_config = dict(
    figsize = (26, 16),
    fontsize = 16, 
    capsize = 3,
    save_fig = True
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for subj in subj_list:

    # Get the data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    if use_test_set:
        dataset = test_dataset 
        dataset_string = 'test'
    else:
        dataset = train_dataset
        dataset_string = 'train'

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Select randomly 4 trial, 1 for each class and plot them (both in time and frequency domain)

    # path = "Saved Results/repetition_hvEEGNet_80/{}/subj {}/recon_error_{}_average.npy".format(dataset_string, subj, 80)
    # recon_error = np.load(path)
    # average_recon_error_per_trial = recon_error.mean(1)

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Create a variable to saved the average spectra for the various channels

    _, tmp_spectra = signal.welch(dataset[0][0].squeeze()[0, :], fs = 250, nperseg = nperseg)
    computed_spectra = np.zeros((len(ch_list), len(dataset), len(tmp_spectra)))

    # Compute the average spectra
    for idx_trial in range(len(dataset)): # Cycle through eeg trials
        x, _ = dataset[idx_trial]

        for i in range(len(ch_list)): # Cycle through channels
            ch = ch_list[i]
            idx_ch = dataset.ch_list == ch
            
            # Compute PSD
            f, x_psd = signal.welch(x.squeeze()[idx_ch, :].squeeze(), fs = 250, nperseg = nperseg)

            computed_spectra[i, idx_trial, :] = x_psd
    
    average_spectra = computed_spectra.mean(1)
    std_spectra = computed_spectra.std(1)

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    # Create figures
    fig_freq, ax_freq = plt.subplots(3, 3, figsize = plot_config['figsize'])

    k = 0
    for i in range(3):
        for j in range(3):
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
            # Plot in frequency domain

            ax_freq[i,j].plot(f, average_spectra[k])
            ax_freq[i,j].fill_between(f, average_spectra[k] + std_spectra[k], average_spectra[k] - std_spectra[k], alpha = 0.25)
            ax_freq[i,j].set_xlabel("Frequency [Hz]")
            ax_freq[i,j].set_ylabel(r"PSD [$\mu V^2/Hz$]")
            ax_freq[i,j].set_xlim([0, 80])
            ax_freq[i,j].legend()
            ax_freq[i,j].grid(True) 
            ax_freq[i,j].set_title(ch_list_for_grid[i][j])
            k += 1

    fig_freq.suptitle("Subject {}".format(subj))
    fig_freq.tight_layout()
    fig_freq.show()


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Save plots

    if plot_config['save_fig']:
        path_save = 'Saved Results/only_original_trial/'
        os.makedirs(path_save, exist_ok = True)

        path_save = 'Saved Results/only_original_trial/'
        path_save += 'grid_{}_average_spectra_S{}_{}'.format(n_trial_to_plot, subj, dataset_string)
        fig_freq.savefig(path_save + ".png", format = 'png')
        fig_freq.savefig(path_save + ".pdf", format = 'pdf')
