"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute the average reconstruction error, std for a single subject and show as an errorplot
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj = 3
ch = 'C3' 
n_trial = 0

use_stft_representation = True
movement_type = 'right' # Use only if use_stft_representation == True

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
