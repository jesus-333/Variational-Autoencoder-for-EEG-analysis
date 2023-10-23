"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py, compute the average reconstruction error and std and create a plot for each subjcet
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt

from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [4]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

plot_config = dict(
    figsize = (10, 8),
    fontsize = 24, 
    capsize = 3,
    use_log_scale = False,
    save_fig = False
)

method_std_computation = 2

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the data and compute average and std recon error

plt.rcParams.update({'font.size': plot_config['fontsize']})
for subj in subj_list:
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    # Compute the average error without NOT skipping the problematic training run
    output = support.compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation = method_std_computation, skip_run = False)
    recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
                label = "All training runs", capsize = plot_config['capsize'],
                color = 'dimgray', linewidth = 1.4,
                marker = "o",
                )

    # Compute the average error without SKIPPING the problematic training run
    output = support.compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation = method_std_computation, skip_run = True)
    recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
                label = "Good training runs", capsize = plot_config['capsize'] + 3,
                color = 'black', linewidth = 2,
                marker = "s", markerfacecolor = 'none', markeredgecolor = 'black', markersize = 11, markeredgewidth = 2
                )

    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Reconstruction Error")
    ax.set_xlabel("Epoch")

    if plot_config['use_log_scale']: ax.set_yscale('log')

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/".format(tot_epoch_training)
        os.makedirs(path_save, exist_ok = True)
        path_save += "average_recon_error_plus_std_subj_{}".format(subj)
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')
