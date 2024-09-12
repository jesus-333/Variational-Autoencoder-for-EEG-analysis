"""
For the subject in subject_list compute the std of each channel of each trials and plot them
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import numpy as np
import matplotlib.pyplot as plt
import os

from library.analysis import support
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3]

plot_config = dict(
    use_TkAgg_backend = True,
    figsize = (12, 6),
    bins = 200,
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
    train_dataset, _, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # Get data (in numpy array)
    train_data = train_dataset.data.numpy().squeeze()
    test_data = test_dataset.data.numpy().squeeze()
    
    # Variable to save std
    std_ch_train = np.zeros((train_data.shape[0], train_data.shape[1]))
    std_ch_test = np.zeros((test_data.shape[0], test_data.shape[1]))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the std for each channel

    for idx_trial in range(train_data.shape[0]):
        for idx_ch in range(train_data.shape[1]):
            std_ch_train[idx_trial, idx_ch] = train_data[idx_trial, idx_ch].std()
            std_ch_test[idx_trial, idx_ch] = test_data[idx_trial, idx_ch].std()

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

    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/d2a_analysis/std_channel/"
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
        path_save = "Saved Results/d2a_analysis/std_channel/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_S{}.png'.format(subj), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')
