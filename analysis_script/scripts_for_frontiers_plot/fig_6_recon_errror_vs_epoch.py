"""
@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute the average reconstruction error, std for a single subject and show as an errorplot
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
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

plot_config = dict(
    figsize = (30, 30),
    fontsize = 24,
    capsize = 3,
    use_log_scale = False,
    save_fig = True
)

method_std_computation = 2
"""
method_std_computation = 1: std along channels and average of std
method_std_computation = 2: meand along channels and std of averages
method_std_computation = 3: std of all the matrix (trials x channels)
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fig, axs = plt.subplots(3, 3, figsize = plot_config['figsize'])
axs = axs.flatten()

for i in range(len(subj_list)) :
    subj = subj_list[i]
    ax = axs[i]
    plt.rcParams.update({'font.size': plot_config['fontsize']})

    # Compute the average error without NOT skipping the problematic training run
    output = support.compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation = method_std_computation, skip_run = False)
    recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj],
                label = "all training runs", capsize = plot_config['capsize'],
                color = 'dimgray', linewidth = 1.4,
                marker = "o",
                )

    # Compute the average error without SKIPPING the problematic training run
    output = support.compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation = method_std_computation, skip_run = True)
    recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj],
                label = "successful runs only", capsize = plot_config['capsize'] + 3,
                color = 'black', linewidth = 2,
                marker = "s", markerfacecolor = 'none', markeredgecolor = 'black', markersize = 17, markeredgewidth = 2
                )

    ax.grid(True)
    if subj == 3: ax.legend(fontsize = plot_config['fontsize'])
    ax.set_ylabel("Reconstruction Error S{}".format(subj), fontsize = plot_config['fontsize'])
    ax.set_xlabel("Epoch no.", fontsize = plot_config['fontsize'])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
    ax.tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'])

    if plot_config['use_log_scale']: ax.set_yscale('log')

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/figure_paper_frontiers/"
    os.makedirs(path_save, exist_ok = True)
    path_save += "FIG_6_recon_error_vs_epoch"
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".jpeg", format = 'jpeg')
    fig.savefig(path_save + ".eps", format = 'eps')
