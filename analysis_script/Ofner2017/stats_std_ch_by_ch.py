"""
For the subject in subject_list compute the std of each channel of each trials and plot them
"""

# TODO Check and complete

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import toml
import numpy as np
import matplotlib.pyplot as plt
import os

from library.dataset import download
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
subj_list = [1]

plot_config = dict(
    use_TkAgg_backend = False,
    figsize = (14, 8),
    bins = 200,
    save_fig = True,
)

path_dataset_config = 'training_scripts/config/Ofner2017/dataset.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]

    # Get dataset
    dataset_config = toml.load(path_dataset_config)
    dataset_config['subjects_list'] = [subj]
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_data, labels_train, ch_list = download.get_Ofner2017(dataset_config, 'train')
    test_data, test_labels, ch_list = download.get_Ofner2017(dataset_config, 'test')
    
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
        ax.grid(True)

    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/Ofner2017/stats_ch/std/"
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
        ax.grid(True)
    
    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/Ofner2017/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_S{}.png'.format(subj), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')
