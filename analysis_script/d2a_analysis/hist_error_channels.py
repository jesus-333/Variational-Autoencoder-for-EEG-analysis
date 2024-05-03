"""
From the tables of the reconstruction error create an histogram for each channel.
The table are the one obtained with the scripts reconstruction_3.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

channel_list = np.asarray(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])
channel_list = np.asarray(['C1'])

plot_config = dict(
    figsize = (20, 8),
    bins = 1000,
    alpha = 0.6,
    use_log_scale = True, # If True use log scale for x axis
    color_train = 'green',
    color_test = 'red',
)


plot_config = dict(
    figsize = (12, 8),
    bins = 1000,
)

tot_epoch_training = 80
epoch_to_plot = 80

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the image

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [2]

# Used for indexing
complet_channel_list = np.asarray(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])

for ch in channel_list:

    idx_ch = complet_channel_list == ch
    recon_error_train = np.zeros((288, len(subj_list)))
    recon_error_test = np.zeros((288, len(subj_list)))

    for i in range(len(subj_list)):
        subj = subj_list[i]

        # Create the array to save the data

        # Load train reconstruction error (session 1)
        path_load_train = 'Saved Results/repetition_hvEEGNet_{}/train/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
        recon_error_train[:, i] = np.load(path_load_train)[:, idx_ch].flatten()

        # Load test reconstruction error (session 2)
        path_load_test = 'Saved Results/repetition_hvEEGNet_{}/test/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
        recon_error_test[:, i] = np.load(path_load_test)[:, idx_ch].flatten()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Create and show the image
    
    # Create the figure
    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    # Plot the data
    for i in range(len(subj_list)):
        subj = subj_list[i]

        axs[0].hist(recon_error_train,  bins = plot_config['bins'],
                label = 'S{}'.format(subj), alpha = plot_config['alpha'])
        axs[1].hist(recon_error_test,  bins = plot_config['bins'],
                label = 'S{}'.format(subj), alpha = plot_config['alpha'])
    
    # Add information to plot
    for ax in axs:
        ax.legend()
        ax.set_xlabel('Reconstruction error')
        if plot_config['use_log_scale']: ax.set_xscale('log')

    axs[0].set_title('Session 1 (Train)')
    axs[1].set_title('Session 2 (Test)')

    fig.suptitle('Channel '.format(ch))
    fig.tight_layout()
    fig.show()
    
    # if plot_config['save_fig']:
    #     # Create pat
    #     path_save = 'Saved Results/d2a_analysis/hist_ch/'
    #     os.makedirs(path_save, exist_ok = True)
    #     
    #     # Save fig
    #     path_save += 'hist_recon_error_ch_{}_bins_{}'.format(ch, plot_config['bins'])
    #     if plot_config['use_log_scale']: path_save += '_LOGSCALE'
    #     fig.savefig(path_save + ".png", format = 'png')
