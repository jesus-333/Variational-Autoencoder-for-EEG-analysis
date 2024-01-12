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

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [6]
ch = 'P1'

use_test_set = True

nperseg = 500

plot_config = dict(
    figsize = (14, 8),
    fontsize = 20, 
    capsize = 3,
    alpha = 0.25,
    color = 'black',
    add_title = False,
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
    train_dataset, validation_dataset, test_dataset , _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    subject_artifacts_map_train = artifacts_map_train[subj_idx]
    subject_artifacts_map_test = artifacts_map_test[subj_idx]

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get average spectra for artifacts/NON artifacts

    idx_ch = train_dataset.ch_list == ch
    
    # For S4 the expert does not find any artifacts in the training set
    average_spectra_NON_artifacts_train, std_spectra_NON_artifacts_train, f = support.compute_average_spectra(train_dataset.data[subject_artifacts_map_train == 0], nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    if subj != 4: 
        average_spectra_artifacts_train, std_spectra_artifacts_train, f = support.compute_average_spectra(train_dataset.data[subject_artifacts_map_train == 1], nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    else:
       average_spectra_artifacts_train, std_spectra_artifacts_train = np.zeros(len(average_spectra_NON_artifacts_train)), np.zeros(len(average_spectra_NON_artifacts_train))

    average_spectra_artifacts_test, std_spectra_artifacts_test, f = support.compute_average_spectra(test_dataset.data[subject_artifacts_map_test == 1], nperseg = nperseg, fs = 250, idx_ch = idx_ch)
    average_spectra_NON_artifacts_test, std_spectra_NON_artifacts_test, f = support.compute_average_spectra(test_dataset.data[subject_artifacts_map_test == 0], nperseg = nperseg, fs = 250, idx_ch = idx_ch)

    average_spectra_list = [average_spectra_artifacts_train, average_spectra_NON_artifacts_train, average_spectra_artifacts_test, average_spectra_NON_artifacts_test]
    std_spectra_list = [std_spectra_artifacts_train, std_spectra_NON_artifacts_train, std_spectra_artifacts_test, std_spectra_NON_artifacts_test]

    label_list = ["Artifacts train", "NON Artifacts train", "Artifacts test", "NON Artifacts test"]

    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -     
    # Create figures
    if plot_config['use_different_figure']:
        fig_freq_1, ax_freq_1 = plt.subplots(1, 1, figsize = plot_config['figsize'])
        fig_freq_2, ax_freq_2 = plt.subplots(1, 1, figsize = plot_config['figsize'])
        fig_freq_3, ax_freq_3 = plt.subplots(1, 1, figsize = plot_config['figsize'])
        fig_freq_4, ax_freq_4 = plt.subplots(1, 1, figsize = plot_config['figsize'])
        
        ax_freq_list = [ax_freq_1, ax_freq_2, ax_freq_3, ax_freq_4]
        fig_freq_list = [fig_freq_1, fig_freq_2, fig_freq_3, fig_freq_4]
    else:
        fig_freq, ax_freq_list = plt.subplots(2, 2, figsize = plot_config['figsize'])
        idx_figures = [[0, 0], [0, 1], [1, 0], [1, 1]] 
        fig_freq_list = [fig_freq]

    for i in range(4):
        average_spectra = average_spectra_list[i]
        std_spectra = std_spectra_list[i]

        if not plot_config['use_different_figure']: ax_freq = ax_freq_list[*idx_figures[i]]
        else : ax_freq = ax_freq_list[i]


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
    
    if plot_config['use_different_figure']:
        for i in range(4):
            fig = fig_freq_list[i]
            label = label_list[i].replace(" ", "_").lower()
            
            fig.tight_layout()
            fig.show()
            
            if plot_config['save_fig']: #TODO eventualmente aggiungere save anche per figura unica
                # Create pat
                path_save = 'Saved Results/d2a_analysis/average_spectra/'
                os.makedirs(path_save, exist_ok = True)
                
                # Save fig
                path_save += 'avg_spectra_S{}_'.format(subj) + label
    
                fig.savefig(path_save + ".png", format = 'png')
                fig.savefig(path_save + ".eps", format = 'eps')
    else:
        fig_freq.tight_layout()
        fig_freq.show()

