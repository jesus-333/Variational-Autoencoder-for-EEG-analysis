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

    path = "Saved Results/repetition_hvEEGNet_80/{}/subj {}/recon_error_{}_average.npy".format(dataset_string, subj, 80)
    recon_error = np.load(path)
    average_recon_error_per_trial = recon_error.mean(1)

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    idx_ch = dataset.ch_list == ch
    average_spectra, std_spectra, f = support.compute_average_spectra(dataset.data, nperseg = nperseg, fs = 250, idx_ch = idx_ch)

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    # Create figures
    fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize'])
            
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Plot in frequency domain

    ax_freq.plot(f, average_spectra,
                 color = plot_config['color'],
                 )
    ax_freq.fill_between(f, average_spectra + std_spectra, average_spectra - std_spectra, 
                         color = plot_config['color'], alpha = plot_config['alpha']
                         )
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$] (S{})".format(subj))
    # ax_freq.legend()
    ax_freq.grid(True) 
    ax_freq.set_xlim([0, 80])
    
    if subj == 1:
        ax_freq.set_ylim([-10, 50])
    elif subj == 2:
        ax_freq.set_ylim([-5, 23])
    elif subj == 5:
        ax_freq.set_ylim([-10, 23])
        
    ax_freq.set_ylim(bottom = -1)

    fig_freq.tight_layout()
    fig_freq.show()


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Save plots

    if plot_config['save_fig']:
        path_save = 'Saved Results/only_original_trial/'
        os.makedirs(path_save, exist_ok = True)

        path_save = 'Saved Results/only_original_trial/'
        path_save += 'single_spectra_average_spectra_S{}_{}'.format(subj, dataset_string)
        fig_freq.savefig(path_save + ".png", format = 'png')
        # fig_freq.savefig(path_save + ".pdf", format = 'pdf')
        fig_freq.savefig(path_save + ".eps", format = 'eps')
