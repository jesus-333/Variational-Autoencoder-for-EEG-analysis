"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute the average reconstruction error, std and variation coefficient
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import matplotlib.pyplot as plt

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 9]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

plot_config = dict(
    figsize = (12, 8),
    fontsize = 14, 
    marker = "o",
    capsize = 3,
    use_log_scale = False,
)
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the data

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
        # Compute the mean and std of the error for each epoch across channels
        for repetition in repetition_list:
            if subj == 4 and (repetition == 4 or repetition == 6): continue
            if subj == 5 and repetition == 19: continue
            
            try:
                path_load = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, subj, epoch, repetition)
                tmp_recon_error = np.load(path_load)
                recon_loss_results_mean[subj][epoch] += tmp_recon_error.mean(1)
                recon_loss_results_std[subj][epoch] += tmp_recon_error.std(1)
            except:
                print("File not found for subj {} - epoch {} - repetition {}".format(subj, epoch, repetition))

        recon_loss_results_mean[subj][epoch] /= len(repetition_list)
        recon_loss_results_std[subj][epoch] /= len(repetition_list)
        # Note that inside recon_loss_results_std[subj][epoch] there are vector of size n_trials

        recon_loss_to_plot_mean[subj].append(recon_loss_results_mean[subj][epoch].mean())
        recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch].mean())

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
plt.rcParams.update({'font.size': plot_config['fontsize']})

for subj in subj_list:
    ax.errorbar(epoch_list, recon_loss_to_plot_mean[subj], yerr = recon_loss_to_plot_std[subj], 
                label = "Subject {}".format(subj), marker = plot_config['marker'], capsize = plot_config['capsize'])

ax.grid(True)
ax.legend()
ax.set_ylabel("Reconstruction Error")
ax.set_xlabel("Epoch")

if plot_config['use_log_scale']: ax.set_yscale('log')

# ax.set_ylim([0, 250])

fig.tight_layout()
fig.show()
