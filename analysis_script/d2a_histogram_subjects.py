"""
From the table of the reconstruction error create an histogram for each subject.
The table are the one obtained with the scripts reconstruction_3.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [2]

plot_config = dict(
    figsize = (12, 8),
    bins = 5,
    color_train = 'green',
    color_test = 'red',
    alpha = 0.6,
    use_log_scale = True, # If True use log scale for x axis
    save_fig = True,
)

tot_epoch_training = 80
epoch_to_plot = 80

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the histogram


for subj in subj_list:
    # Create the array to save the data

    # Load train reconstruction error (session 1)
    path_load_train = 'Saved Results/repetition_hvEEGNet_{}/train/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_train = np.load(path_load_train).flatten()

    # Load test reconstruction error (session 2)
    path_load_test = 'Saved Results/repetition_hvEEGNet_{}/test/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_test = np.load(path_load_test).flatten()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot the histogram

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    ax.hist(recon_error_test,  bins = plot_config['bins'],
            color = plot_config['color_test'], label = 'Session 2 (Test)', alpha = 0.5)
    ax.hist(recon_error_train,  bins = plot_config['bins'],
            color = plot_config['color_train'], label = 'Session 1 (Train)', alpha = 0.5)

    ax.legend()
    ax.set_title('Subject {}'.format(subj))
    ax.set_xlabel('Reconstruction error')
    if plot_config['use_log_scale']: ax.set_xscale('log')

    fig.tight_layout()
    fig.show()


    if plot_config['save_fig']:
        # Create pat
        path_save = 'Saved Results/d2a_analysis/hist_subj/'
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'hist_recon_error_S{}_bins_{}'.format(subj, plot_config['bins'])
        if plot_config['use_log_scale']: path_save += '_LOGSCALE'
        fig.savefig(path_save + ".png", format = 'png')
