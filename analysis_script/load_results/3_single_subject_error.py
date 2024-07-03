"""
Created on Fri Sep  1 10:03:59 2023

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

from library.config import config_plot as cp
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [4]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

plot_config = dict(
    figsize = (12, 8),
    fontsize = 16, 
    capsize = 3,
    use_log_scale = False,
    save_fig = False
)

method_std_computation = 2
"""
method_std_computation = 1: std along channels and average of std
method_std_computation = 2: meand along channels and std of averages
method_std_computation = 3: std of all the matrix (trials x channels)
"""

line_config_per_subject = cp.get_style_per_subject_error_plus_std()

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the data and compute average and std recon error

recon_loss_results_mean = dict() # Save for each subject/repetition/epoch the average reconstruction error across channels
recon_loss_results_std = dict() # Save for each subject/repetition/epoch the std of the reconstruction error across channels

recon_loss_to_plot_mean = dict()
recon_loss_to_plot_std = dict()

for subj in subj_list:
    recon_loss_results_mean[subj] = dict()
    recon_loss_results_std[subj] = dict()
    recon_loss_to_plot_mean[subj] = list()
    recon_loss_to_plot_std[subj] = list()

    for epoch in epoch_list:
        recon_loss_results_mean[subj][epoch] = 0
        recon_loss_results_std[subj][epoch] = 0
        
        valid_repetition = 0
        
        # Compute the mean and std of the error for each epoch across channels
        for repetition in repetition_list:
            if support.skip_training_run(subj, repetition):
                print("Skip run {} subj {}".format(repetition, subj))
                continue
            
            try:
                path_load = 'Saved Results/repetition_hvEEGNet_{}/subj {}/train/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, subj, epoch, repetition)
                tmp_recon_error = np.load(path_load)
                
                recon_loss_results_mean[subj][epoch] += tmp_recon_error.mean(1)
                if method_std_computation == 1:             
                    recon_loss_results_std[subj][epoch] += tmp_recon_error.std(1)
                elif method_std_computation == 2:
                    recon_loss_results_std[subj][epoch] += tmp_recon_error.mean(1)
                elif method_std_computation == 3:
                    recon_loss_results_std[subj][epoch] += tmp_recon_error.std()
                
                valid_repetition += 1
            except:
                print("File not found for subj {} - epoch {} - repetition {}".format(subj, epoch, repetition))

            # if epoch == 80:
            #     print("Epoch {} - rep {}".format(epoch, repetition))
            #     print("std:", tmp_recon_error.std())
            #     print("- - - - -\n")

        recon_loss_results_mean[subj][epoch] /= valid_repetition
        recon_loss_results_std[subj][epoch] /= valid_repetition
        # Note that inside recon_loss_results_std[subj][epoch] there are vector of size n_trials
        
        recon_loss_to_plot_mean[subj].append(recon_loss_results_mean[subj][epoch].mean())
        if method_std_computation == 1:
            recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch].mean())
        elif method_std_computation == 2:
            recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch].std())
        elif method_std_computation == 3:
            recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch])

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
for subj in subj_list:
    plt.rcParams.update({'font.size': plot_config['fontsize']})

    subj_key = "subj_{}".format(subj)
    line_config = line_config_per_subject[subj_key]
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
                capsize = plot_config['capsize'],
                marker = line_config['marker'], color = line_config['color'], linestyle = line_config['linestyle']
                )

ax.grid(True)
ax.legend()
ax.set_ylabel("Reconstruction Error")
ax.set_xlabel("Epoch")

if plot_config['use_log_scale']: ax.set_yscale('log')
# ax.set_ylim([0, 250])

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/repetition_hvEEGNet_{}/".format(tot_epoch_training)
    os.makedirs(path_save, exist_ok = True)
    path_save += "average_recon_error_plus_std_subj_{}".format(subj)
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".pdf", format = 'pdf')
