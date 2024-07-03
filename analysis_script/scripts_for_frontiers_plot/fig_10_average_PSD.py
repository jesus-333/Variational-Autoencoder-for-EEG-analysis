"""
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

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 5]
ch = 'Cz'

use_test_set = True

nperseg = 500

plot_config = dict(
    figsize = (15, 20),
    fontsize = 18,
    capsize = 3,
    alpha = 0.25,
    color = 'black',
    save_fig = True
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
plt.rcParams.update({'font.size': plot_config['fontsize']})

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create figures
fig_freq, axs = plt.subplots(3, 2, figsize = plot_config['figsize'])

for i in range(len(subj_list)) :
    subj = subj_list[i]

    ax_freq_1 = axs[i, 0]
    ax_freq_2 = axs[i, 1]

    ax_freq_list = [ax_freq_1, ax_freq_2]

    # Get the data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in frequency domain (Train data)

    idx_ch = train_dataset.ch_list == ch
    average_spectra, std_spectra, f = support.compute_average_spectra(train_dataset.data, nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    ax_freq_1.plot(f, average_spectra, color = plot_config['color'])
    ax_freq_1.fill_between(f, average_spectra + std_spectra, average_spectra - std_spectra,
                           color = plot_config['color'], alpha = plot_config['alpha']
                           )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in frequency domain (Test data)

    idx_ch = train_dataset.ch_list == ch
    average_spectra, std_spectra, f = support.compute_average_spectra(test_dataset.data, nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    ax_freq_2.plot(f, average_spectra, color = plot_config['color'])
    ax_freq_2.fill_between(f, average_spectra + std_spectra, average_spectra - std_spectra,
                           color = plot_config['color'], alpha = plot_config['alpha']
                           )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for ax_freq in ax_freq_list :
        ax_freq.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
        ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$] (S{})".format(subj), fontsize = plot_config['fontsize'])
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

        ax_freq.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
        ax_freq.tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'])

fig_freq.tight_layout()
fig_freq.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save plots

if plot_config['save_fig']:
    path_save = "Saved Results/figure_paper_frontiers/"
    os.makedirs(path_save, exist_ok = True)
    path_save += "FIG_10_average_PSD"
    fig_freq.savefig(path_save + ".png", format = 'png')
    fig_freq.savefig(path_save + ".jpeg", format = 'jpeg')
    fig_freq.savefig(path_save + ".eps", format = 'eps')
