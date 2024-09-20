"""
For the subject in subject_list compute the std of each channel of each trials and plot them
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import numpy as np
import matplotlib.pyplot as plt
import os

from library.dataset import download
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3]
subj_list = [1]

plot_config = dict(
    use_TkAgg_backend = True,
    figsize = (12, 6),
    bins = 400,
    xlim = [0, 200],
    use_log_scale_x_axis = False,
    use_log_scale_y_axis = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]

    # Get dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_data, labels_train, ch_list = download.get_BI2014a(dataset_config, 'train')
    test_data, test_labels, ch_list = download.get_BI2014a(dataset_config, 'test')
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the std for each channel

    std_ch_train = train_data.std(2)
    std_ch_test = test_data.std(2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the data (histogram)

    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    axs[0].hist(std_ch_train.flatten(), bins = plot_config['bins'], color = 'black')
    axs[0].set_title('Train data')

    axs[1].hist(std_ch_test.flatten(), bins = plot_config['bins'], color = 'black')
    axs[1].set_title('Test data')

    for ax in axs:
        ax.set_xlabel('Standard deviation')
        ax.set_ylabel('Number of occurrences')
        if 'xlim' in plot_config : ax.set_xlim(plot_config['xlim'])
        if plot_config['use_log_scale_x_axis'] : ax.set_xscale('log')
        if plot_config['use_log_scale_y_axis'] : ax.set_yscale('log')

    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/BI2014a/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'hist_std_ch_by_ch_S{}.png'.format(subj), format = 'png')
        # fig.savefig(path_save + 'hist_std_ch_by_ch_S{}.pdf'.format(subj), format = 'pdf')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the data (average per trial) 

    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    axs[0].plot(std_ch_train.mean(axis = 1), color = 'black')
    axs[0].fill_between(np.arange(std_ch_train.shape[0]), std_ch_train.mean(axis = 1) - std_ch_train.std(axis = 1), std_ch_train.mean(axis = 1) + std_ch_train.std(axis = 1), color = 'black', alpha = 0.2)
    axs[0].set_title('Train data')

    axs[1].plot(std_ch_test.mean(axis = 1), color = 'black')
    axs[1].fill_between(np.arange(std_ch_test.shape[0]), std_ch_test.mean(axis = 1) - std_ch_test.std(axis = 1), std_ch_test.mean(axis = 1) + std_ch_test.std(axis = 1), color = 'black', alpha = 0.2)
    axs[1].set_title('Test data')

    for ax in axs:
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Average standard deviation per trial')
    
    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/BI2014a/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_S{}.png'.format(subj), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')
