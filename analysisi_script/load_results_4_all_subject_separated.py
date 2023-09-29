"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute the average reconstruction error, std and variation coefficient
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

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_average_and_std_reconstruction_error(subj_list, epoch_list, repetition_list, method_std_computation = 1):
    """
    method_std_computation = 1: std along channels and average of std
    method_std_computation = 2: meand along channels and std of averages
    method_std_computation = 3: std of all the matrix (trials x channels)

    """
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
                if subj == 4 and (repetition == 4 or repetition == 6): continue
                if subj == 5 and repetition == 19: continue
                if subj == 8 and repetition == 12: continue
                
                try:
                    path_load = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, subj, epoch, repetition)
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

    return recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

plot_config = dict(
    figsize = (20, 20),
    fontsize = 16, 
    capsize = 3,
    use_log_scale = False,
    save_fig = False
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the data and compute average and std recon error


fig, ax = plt.subplots(3, 3, figsize = plot_config['figsize'])
plt.rcParams.update({'font.size': plot_config['fontsize']})

idx_subj = 0
for i in range(3):
    for j in range(3):
        subj = subj_list[idx_subj]
        idx_subj += 1
        subj_key = "subj_{}".format(subj)

        # output = compute_average_and_std_reconstruction_error(subj_list, epoch_list, repetition_list, method_std_computation = 1)
        # recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
        # ax[i, j].errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
        #             label = "Method 1", capsize = plot_config['capsize'],
                    # )


        # output = compute_average_and_std_reconstruction_error(subj_list, epoch_list, repetition_list, method_std_computation = 2)
        # recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
        # ax[i, j].errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
        #             label = "Method 2", capsize = plot_config['capsize'],
        #             )

        output = compute_average_and_std_reconstruction_error(subj_list, epoch_list, repetition_list, method_std_computation = 3)
        recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output
        ax[i, j].errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
                    label = "Method 3", capsize = plot_config['capsize'],
                    )

        ax[i, j].grid(True)
        ax[i, j].legend()
        ax[i, j].set_ylabel("Reconstruction Error")
        ax[i, j].set_xlabel("Epoch")

        if plot_config['use_log_scale']: ax[i, j].set_yscale('log')

# ax[i, j].set_ylim([0, 250])

fig.tight_layout()
fig.show()

# if plot_config['save_fig']:
#     path_save = "Saved Results/repetition_hvEEGNet_{}/".format(tot_epoch_training)
#     os.makedirs(path_save, exist_ok = True)
#     path_save += "average_recon_error_plus_std"
#     fig.savefig(path_save + ".png", format = 'png')
#     fig.savefig(path_save + ".pdf", format = 'pdf')
