"""
Create an histogram with all the data of all the trials and all the channels for a specific subject.

Optionally you can select only a specific channel
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import os

import numpy as np
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [3]
use_test_set = False

channel_to_plot = None

distribution_type = 1 # 1 means normalize automatically through matplotlib. Create a continuos PDF (i.e. the integral of the area under the histogram is equal to 1)
distribution_type = 2 # 2 means the creation of a discrete PDF ( i.e. the hights of the bins is divided by the total number of the samples )

plot_config = dict(
    figsize = (10, 8),
    use_same_plot = True,
    bins = 50,
    linewidth = 1.5,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if plot_config['use_same_plot']: fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

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
    else:
        dataset = train_dataset
        dataset_string = 'train'
    
    # Flatten the data and (OPTIONAL) select a channel
    if channel_to_plot is not None:
        idx_ch = dataset.ch_list == channel_to_plot
        data = dataset.data.squeeze()[:, idx_ch, :].flatten()
    else:
        data = dataset.data.flatten()

    # Create a figure for each plot
    if not plot_config['use_same_plot']: fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    
    # Plot the data
    if distribution_type == 1:
        ax.hist(data.sort()[0], plot_config['bins'], density = True,
                label = 'S{}'.format(subj), histtype = 'step', linewidth = plot_config['linewidth']
                )

        ax.set_ylabel("Continuos PDF")
    elif distribution_type == 2:
        p_x, bins_position = np.histogram(data.sort()[0], bins = plot_config['bins'], density = False)
        p_x = p_x / len(data)

        step_bins = bins_position[1] - bins_position[0]
        bins_position = bins_position[1:] - step_bins

        ax.step(bins_position, p_x,
                label = 'S{}'.format(subj),
                )

        # ax.fill_between(bins_position, p_x,
        #                 alpha = 0.4, step = 'pre'
        #                 )
        ax.set_ylabel("Discrete PDF")
    else:
        ax.hist(data.sort()[0], bins = plot_config['bins'], label = 'S{}'.format(subj), histtype = 'step', linewidth = 1.5)

    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r"Amplitude [$\mu$V]")
    ax.set_title("S{} - {}".format(subj, dataset_string))
    ax.set_title("S{}".format(subj))
    ax.set_xlim([-8, 8])
    ax.set_ylim([0, 650])
    
    if plot_config['use_log_scale_x']: ax.set_xscale('log')
    if plot_config['use_log_scale_y']: ax.set_yscale('log')

    if plot_config['save_fig'] and not plot_config['use_same_plot']:
        ax.set_title("S{} - {}".format(subj, dataset_string))
        fig.tight_layout()
        fig.show()

        # Create pat
        path_save = 'Saved Results/d2a_analysis/hist_samples/bins {}/'.format(plot_config['bins'])
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'hist_samples_S{}_{}_bins_{}'.format(subj, dataset_string, plot_config['bins'])
        if plot_config['use_log_scale_x']: path_save += '_LOGSCALE_X'
        if plot_config['use_log_scale_y']: path_save += '_LOGSCALE_Y'
        fig.savefig(path_save + ".png", format = 'png')
        # fig.savefig(path_save + ".eps", format = 'eps')

if plot_config['save_fig'] and plot_config['use_same_plot']:
    fig.tight_layout()
    fig.show()

    # Create pat
    path_save = 'Saved Results/d2a_analysis/hist_samples/bins {}/'.format(plot_config['bins'])
    os.makedirs(path_save, exist_ok = True)
    
    # Save fig
    path_save += 'hist_samples_all_subj_{}_bins_{}'.format(dataset_string, plot_config['bins'])
    if plot_config['use_log_scale_x']: path_save += '_LOGSCALE_X'
    if plot_config['use_log_scale_y']: path_save += '_LOGSCALE_Y'
    fig.savefig(path_save + ".png", format = 'png')
    # fig.savefig(path_save + ".eps", format = 'eps')
