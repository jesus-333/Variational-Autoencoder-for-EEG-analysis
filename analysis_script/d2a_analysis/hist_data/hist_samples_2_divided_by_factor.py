"""
Divide the samples by class/chanel/artifacts and plot their distrbution
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [3]
use_test_set = False

factor_for_division = 'class'
# factor_for_division = 'channel'
factor_for_division = 'artifacts'

ch_to_plot = None
ch_to_plot = ['C3', 'Fz', 'POz']

plot_config = dict(
    # figsize = (16, 10),
    figsize = (25, 15),
    bins = 100,
    normalize = True,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subj {}".format(subj))

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # select train/test data
    if use_test_set:
        dataset = test_dataset
        dataset_string = 'test'
        artifacts_map = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_test.csv').T.to_numpy()[1:, :][i, :] # N.b. the first line contains simply the index of the samples. The other lines containts the map of the artifacts
    else:
        dataset = train_dataset
        dataset_string = 'train'
        artifacts_map = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_train.csv').T.to_numpy()[1:, :][i, :]
    
    # Variable to save the data divided by factor and numpy array of the data
    data_divided = []
    data = dataset.data.squeeze()
    
    # Divide the data
    if factor_for_division == 'channel': # Divide the data by channel
        if ch_to_plot is None:
            ch_list = dataset.ch_list
        else:
            ch_list = ch_to_plot

        labels_for_plot = ch_list
        for j in range(len(ch_list)):
            idx_ch = ch_list[j] == dataset.ch_list
            tmp_ch_data = data[:, idx_ch, :].flatten()
            data_divided.append(tmp_ch_data)
    
    elif factor_for_division == 'class': # Divide the data by class (label)
        labels = dataset.labels
        labels_for_plot = np.unique(labels)

        for j in range(len(labels_for_plot)):
            label = labels_for_plot[j]
            idx_label = labels == label
            tmp_label_data = data[idx_label, :, :]
            data_divided.append(tmp_label_data.flatten())

        label_dict = {0 : 'left hand', 1 : 'right hand', 2 : 'foot', 3 : 'tongue'}
        labels_for_plot = [label_dict[label] for label in labels_for_plot]

    elif factor_for_division == 'artifacts':
        labels_for_plot = ['Artifacts', 'NON artifacts']
        data_divided.append(data[artifacts_map, :, :].flatten())
        data_divided.append(data[np.logical_not(artifacts_map), :, :].flatten())

    else:
        raise ValueError("factor_for_division must have one of the following values: channel, class or artifacts")
    
    # Create the plot figure for the plot
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    
    # Plot the data
    for j in range(len(data_divided)):
        ax.hist(data_divided[j], bins = plot_config['bins'], density = plot_config['normalize'],
                label = labels_for_plot[j], alpha = 0.8,
                histtype = 'step', linewidth = 1.5,
                )

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"Amplitude [$\mu$V]")
    ax.set_title("S{} - {}".format(subj, dataset_string))

    if plot_config['bins'] >= 300 and factor_for_division == 'channel':
        if plot_config['normalize']: ax.set_ylim([0, 0.06])
        else: ax.set_ylim([0, 8000])
        ax.set_xlim([-110, 110])

    if plot_config['bins'] >= 300 and factor_for_division == 'class':
        ax.set_xlim([-110, 110])
        if plot_config['normalize']: ax.set_ylim([0, 0.055])
        else: ax.set_ylim([0, 53000])
    
    if plot_config['use_log_scale_x']: ax.set_xscale('log')
    if plot_config['use_log_scale_y']: ax.set_yscale('log')

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        # Create pat
        path_save = 'Saved Results/d2a_analysis/hist_samples/bins {}/'.format(plot_config['bins'])
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'hist_samples_S{}_{}_bins_{}_factor_{}'.format(subj, dataset_string, plot_config['bins'], factor_for_division)
        if plot_config['use_log_scale_x']: path_save += '_LOGSCALE_X'
        if plot_config['use_log_scale_y']: path_save += '_LOGSCALE_Y'
        fig.savefig(path_save + ".png", format = 'png')
        # fig.savefig(path_save + ".eps", format = 'eps')
