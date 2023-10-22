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

subj = 6
ch_list = [['FC3', 'Fz', 'FC4'], ['C5', 'Cz', 'C6'], ['P1', 'POz', 'P2']]

use_test_set = True

n_trial_to_plot = 3
 
t_min = 2
t_max = 4

nperseg = 500

plot_config = dict(
    figsize = (26, 16),
    fontsize = 16, 
    capsize = 3,
    save_fig = True
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the data
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

if use_test_set:
    dataset = test_dataset 
    dataset_string = 'test'
else:
    dataset = train_dataset
    dataset_string = 'train'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Select randomly 4 trial, 1 for each class and plot them (both in time and frequency domain)

path = "Saved Results/repetition_hvEEGNet_80/{}/subj {}/recon_error_{}_average.npy".format(dataset_string, subj, 80)
recon_error = np.load(path)
average_recon_error_per_trial = recon_error.mean(1)

# Sort trial by recon error so it it is possible to take the n trials with the worst reconstruction
idx_sorted = np.argsort(np.argsort(average_recon_error_per_trial)) 

# Create figures
fig_time, ax_time = plt.subplots(3, 3, figsize = plot_config['figsize'])
fig_freq, ax_freq = plt.subplots(3, 3, figsize = plot_config['figsize'])

# Plot the trials
for n in range(n_trial_to_plot):
    x, _ = dataset[idx_sorted[n]]
    
    # Select section of the trial to visualize and create time vector
    tmp_t = np.linspace(2, 6, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    
    for i in range(3):
        for j in range(3):
            
            ch_to_plot = ch_list[i][j]
            idx_ch = train_dataset.ch_list == ch_to_plot
            
            # Select channel and time samples
            x_time = x.squeeze()[idx_ch, idx_t]
            
            # Compute PSD
            f, x_psd = signal.welch(x.squeeze()[idx_ch, :].squeeze(), fs = 250, nperseg = nperseg)
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
            # Plot in time domain

            ax_time[i,j].plot(t, x_time, label = "trial {}".format(idx_sorted[n]))
            ax_time[i,j].set_xlabel("Time [s]")
            ax_time[i,j].set_ylabel(r"Amplitude [$\mu$V]")
            ax_time[i,j].set_xlim([t_min, t_max])
            ax_time[i,j].legend()
            ax_time[i,j].grid(True)
            ax_time[i,j].set_title(ch_to_plot)
    
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
            # Plot in frequency domain

            ax_freq[i,j].plot(f, x_psd, label = "trial {}".format(idx_sorted[n]))
            ax_freq[i,j].set_xlabel("Frequency [Hz]")
            ax_freq[i,j].set_ylabel(r"PSD [$\mu V^2/Hz$]")
            ax_freq[i,j].set_xlim([0, 80])
            ax_freq[i,j].legend()
            ax_freq[i,j].grid(True) 
            ax_freq[i,j].set_title(ch_to_plot)

    fig_freq.tight_layout()
    fig_freq.show()
    fig_time.tight_layout()
    fig_time.show()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Save plots

if plot_config['save_fig']:
    path_save = 'Saved Results/only_original_trial/'
    os.makedirs(path_save, exist_ok = True)
    path_save += 'grid_{}_worst_trial_S{}_{}_time'.format(n_trial_to_plot, subj, dataset_string)
    fig_time.savefig(path_save + ".png", format = 'png')
    fig_time.savefig(path_save + ".pdf", format = 'pdf')

    path_save = 'Saved Results/only_original_trial/'
    path_save += 'grid_{}_worst_trial_S{}_{}_freq'.format(n_trial_to_plot, subj, dataset_string)
    fig_freq.savefig(path_save + ".png", format = 'png')
    fig_freq.savefig(path_save + ".pdf", format = 'pdf')
