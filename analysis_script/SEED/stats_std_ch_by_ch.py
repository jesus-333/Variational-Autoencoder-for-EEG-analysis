"""
For the subject in subject_list compute the std of each channel of each trials and plot them
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import numpy as np
import matplotlib.pyplot as plt
import os
import toml

from library.analysis import support
from library.dataset import preprocess
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [2, 3]
subj_list = np.arange(15) + 1
file_path = 'data/SEED/'
trials_length_in_seconds = 4

plot_config = dict(
    use_TkAgg_backend = False,
    figsize = (12, 6),
    bins = 200,
    use_log_scale = False,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]

    # Get all the file for a single subject
    filename_list = []
    list_files = os.listdir(file_path)
    for file in list_files :
        if '_' in file :
            file_id = int(file.split('_')[0])
            if file_id == subj : filename_list.append(file_path + file)
        else :
            continue

    # Get subject data and model
    dataset_config = toml.load('training_scripts/config/SEED/dataset.toml')
    train_data, test_data, _ = preprocess.get_dataset_SEED_split_in_train_test_validation(filename_list[0], trials_length_in_seconds,
                                                                                          dataset_config['percentage_split_train_test'], dataset_config['percentage_split_train_validation']
                                                                                          )
    
    # Get data (in numpy array)
    train_data = train_data.squeeze()
    test_data = test_data.squeeze()

    # train_data *= 40
    
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
        if plot_config['use_log_scale'] : ax.set_yscale('log')

    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/SEED/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        if plot_config['use_log_scale'] : 
            path_save = "Saved Results/SEED/stats_ch/std/hist_mean_ch_by_ch_S{}_log".format(subj)
        else :
            path_save = "Saved Results/SEED/stats_ch/std/hist_mean_ch_by_ch_S{}".format(subj)

        fig.savefig(path_save + '.png', format = 'png')
        # fig.savefig(path_save + '.pdf', format = 'pdf')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the data (average per trial) 

    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])
    
    # Train set
    axs[0].plot(std_ch_train.mean(axis = 1), color = 'black')
    axs[0].fill_between(np.arange(std_ch_train.shape[0]), std_ch_train.mean(axis = 1) - std_ch_train.std(axis = 1), std_ch_train.mean(axis = 1) + std_ch_train.std(axis = 1), color = 'black', alpha = 0.2)
    axs[0].set_title('Train data')
    
    # Test set
    axs[1].plot(std_ch_test.mean(axis = 1), color = 'black')
    axs[1].fill_between(np.arange(std_ch_test.shape[0]), std_ch_test.mean(axis = 1) - std_ch_test.std(axis = 1), std_ch_test.mean(axis = 1) + std_ch_test.std(axis = 1), color = 'black', alpha = 0.2)
    axs[1].set_title('Test data')
    
    # Add other info
    for ax in axs:
        # Labels
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Average standard deviation per trial')
    
    fig.suptitle('Subject {}'.format(subj))
    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/SEED/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_S{}.png'.format(subj), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')
