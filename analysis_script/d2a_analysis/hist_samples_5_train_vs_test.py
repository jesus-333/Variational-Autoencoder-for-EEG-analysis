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
# subj_list = [2, 5]

channel_to_plot = None

distribution_type = 1 # 1 means normalize automatically through matplotlib. Create a continuos PDF (i.e. the integral of the area under the histogram is equal to 1)
distribution_type = 2 # 2 means the creation of a discrete PDF ( i.e. the hights of the bins is divided by the total number of the samples )

plot_config = dict(
    figsize = (10, 5),
    bins = 200,
    linewidth = 1.5,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 18,
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
    
    # Flatten the data and (OPTIONAL) select a channel
    if channel_to_plot is not None:
        idx_ch = train_dataset.ch_list == channel_to_plot
        train_data = train_dataset.data.squeeze()[:, idx_ch, :].flatten()
        test_data = test_dataset.data.squeeze()[:, idx_ch, :].flatten()
    else:
        train_data = train_dataset.data.flatten()
        test_data = test_dataset.data.flatten()

    # Create a figure for each plot
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    
    # Plot the data
    if distribution_type == 1:
        ax.hist(train_data.sort()[0], plot_config['bins'], density = True,
                label = 'S{} Train'.format(subj), histtype = 'step', linewidth = plot_config['linewidth']
                )

        ax.hist(test_data.sort()[0], plot_config['bins'], density = True,
                label = 'S{} Test'.format(subj), histtype = 'step', linewidth = plot_config['linewidth']
                )

        ax.set_ylabel("Continuos PDF")
    elif distribution_type == 2:
        train_p_x, bins_position = np.histogram(train_data.sort()[0], bins = plot_config['bins'], density = False)
        train_p_x = train_p_x / len(train_data)
        step_bins = bins_position[1] - bins_position[0]
        bins_position = bins_position[1:] - step_bins
        ax.step(bins_position, train_p_x,
                label = 'Train', color = 'green', linewidth = plot_config['linewidth']
                )

        test_p_x, bins_position = np.histogram(test_data.sort()[0], bins = plot_config['bins'], density = False)
        test_p_x = test_p_x / len(test_data)
        step_bins = bins_position[1] - bins_position[0]
        bins_position = bins_position[1:] - step_bins
        ax.step(bins_position, test_p_x,
                label = 'Test', color = 'red', linewidth = plot_config['linewidth'], 
                )

        ax.set_ylabel("Discrete PDF", fontsize = plot_config['fontsize'])
    else:
        ax.hist(train_data.sort()[0], bins = plot_config['bins'], label = 'S{}'.format(subj), histtype = 'step', linewidth = 1.5)
        ax.hist(test_data.sort()[0], bins = plot_config['bins'], label = 'S{}'.format(subj), histtype = 'step', linewidth = 1.5)

    ax.grid(True)
    ax.legend(fontsize = plot_config['fontsize'])
    ax.set_xlabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])
    # ax.set_title("S{} - Hist ALL samples".format(subj))
    # ax.set_title("S{}".format(subj))
    ax.set_xlim([-50, 50])
    # ax.set_ylim([0, 650])
    ax.tick_params(axis = 'both', labelsize = plot_config['fontsize'])
    
    if plot_config['use_log_scale_x']: ax.set_xscale('log')
    if plot_config['use_log_scale_y']: ax.set_yscale('log')

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:

        # Create pat
        path_save = 'Saved Results/d2a_analysis/hist_samples_train_vs_test/bins {}/'.format(plot_config['bins'])
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'hist_samples_S{}_train_vs_test_bins_{}'.format(subj, plot_config['bins'])
        if plot_config['use_log_scale_x']: path_save += '_LOGSCALE_X'
        if plot_config['use_log_scale_y']: path_save += '_LOGSCALE_Y'
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')
