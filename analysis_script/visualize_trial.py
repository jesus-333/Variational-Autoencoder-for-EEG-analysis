"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua
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

subj = 3
ch = 'C3' # Use only if use_stft_representation == True

use_stft_representation = False
movement_type = 'right' # Use only if use_stft_representation == True

select_random_trial = False # Only valid if use_stft_representation == False
trial_list = [0, 9] # Used only if  select_random_trial == False

t_min = 2
t_max = 4

nperseg = 500

plot_config = dict(
    figsize = (12, 8),
    fontsize = 16, 
    capsize = 3,
    cmap = 'plasma',
    save_fig = True
)

# Prendere c3 per la mano destra e c4 per la mano sinistra

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the data
dataset_config = cd.get_moabb_dataset_config([subj], use_stft_representation)
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Show the average trial for each class through the stft
if use_stft_representation:
    average_trial, vmin, vmax = train_dataset.get_average_trial(ch, movement_type)

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    im = ax.pcolormesh(train_dataset.t, train_dataset.f, average_trial,
                       shading = 'gouraud', cmap = plot_config['cmap'],
                       vmin = vmin, vmax = vmax)

    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')

    ax.set_ylim([0, 80])

    fig.colorbar(im)
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = 'Saved Results/stft/'
        os.makedirs(path_save, exist_ok = True)
        path_save += 'subj_{}_ch_{}_class_{}'.format(subj, ch, movement_type)
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot trial in time domain
else:
    trials_dict = {0 : None, 1 : None, 2 : None, 3 : None}
    label_dict = {0 : 'left hand', 1 : 'right hand', 2 : 'foot', 3 : 'tongue' }
    label_to_ch = {0 : 'C4', 1 : 'C3', 2 : 'Fz', 3 : 'Fz' }
    idx_trials_saved = []
    
    n_trials = 0
    label_trials_saved = []
    
    if select_random_trial: # Get 4 random trial of different class  
        idx_trials = np.arange(len(train_dataset))
    else: # Save the specified trial
        idx_trials = trial_list
        
    for i in range(len(train_dataset)):
        trial, label = train_dataset[idx_trials[i]]
        label = int(label)

        if trials_dict[label] is None:
            trials_dict[label] = trial
            n_trials += 1
            idx_trials_saved.append(idx_trials[i] + 1)
            label_trials_saved.append(label)

        if select_random_trial == True and n_trials == 4: break 
        if select_random_trial == False and n_trials == len(trial_list): break
    
    fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize'])
    fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize'])
    for i in range(n_trials):
        x = trials_dict[label_trials_saved[i]]
        
        # Select section of the trial to visualize and create time vector
        tmp_t = np.linspace(2, 6, x.shape[-1])
        idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
        t = tmp_t[idx_t]
        idx_ch = train_dataset.ch_list == label_to_ch[i]
        label_name = label_dict[int(label)]
        
        # Select channel and time samples
        x = x.squeeze()[idx_ch, idx_t]
        
        # Compute PSD
        f, x_psd = signal.welch(x, fs = 250, nperseg = nperseg)
        
        # Plot in time domain

        ax_time.plot(t, x, 
                     label = "trial {} - ch {} - {} movement".format(idx_trials_saved[i], label_to_ch[label_trials_saved[i]], label_dict[label_trials_saved[i]])
                     )
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel(r"Amplitude [$\mu$V]")
        ax_time.set_xlim([t_min, t_max])
        ax_time.legend()
        ax_time.grid(True)
        
        fig_time.tight_layout()
        fig_time.show()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Plot in frequency domain

        ax_freq.plot(f, x_psd,
                     label = "trial {} - ch {} - {} movement".format(idx_trials_saved[i], label_to_ch[label_trials_saved[i]], label_dict[label_trials_saved[i]])
                     )
        ax_freq.set_xlabel("Frequency [Hz]")
        ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$]")
        ax_freq.set_xlim([0, 50])
        ax_freq.legend()
        ax_freq.grid(True)
        
        fig_freq.tight_layout()
        fig_freq.show()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Save plots

        if plot_config['save_fig']:
            path_save = 'Saved Results/only_original_trial/'
            os.makedirs(path_save, exist_ok = True)
            path_save += 'example_trial_for_paper_time'
            fig_time.savefig(path_save + ".png", format = 'png')
            fig_time.savefig(path_save + ".pdf", format = 'pdf')

            path_save = 'Saved Results/only_original_trial/'
            path_save += 'example_trial_for_paper_freq'
            fig_freq.savefig(path_save + ".png", format = 'png')
            fig_freq.savefig(path_save + ".pdf", format = 'pdf')
