#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np

from library.config import config_plot as cp
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
epoch_list = [80]

use_test_set = False

plot_config = dict(
    figsize = (12, 8),
    fontsize = 16, 
    capsize = 3,
    use_log_scale = False,
    save_fig = True
)

method_std_computation = 2
skip_run = True
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

recon_loss_average_mean = dict()
recon_loss_average_std = dict()

if use_test_set: string_dataset = 'test'
else: string_dataset = 'train'

for subj in subj_list:
    recon_loss_results_mean[subj] = dict()
    recon_loss_results_std[subj] = dict()
    recon_loss_average_mean[subj] = list()
    recon_loss_average_std[subj] = list()
    
    for epoch in epoch_list:
        recon_loss_results_mean[subj][epoch] = 0
        recon_loss_results_std[subj][epoch] = 0
        
        valid_repetition = 0
        
        # Compute the mean and std of the error for each epoch across channels
        for repetition in repetition_list:
            if support.skip_training_run(subj, repetition) and skip_run:
                print("Skip run {} subj {}".format(repetition, subj))
                continue
            
            try:
                path_load = 'Saved Results/repetition_hvEEGNet_{}/{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, string_dataset, subj, epoch, repetition)
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
        
        recon_loss_average_mean[subj].append(recon_loss_results_mean[subj][epoch].mean())
        if method_std_computation == 1:
            recon_loss_average_std[subj].append(recon_loss_results_std[subj][epoch].mean())
        elif method_std_computation == 2:
            recon_loss_average_std[subj].append(recon_loss_results_std[subj][epoch].std())
        elif method_std_computation == 3:
            recon_loss_average_std[subj].append(recon_loss_results_std[subj][epoch])

#%% Create array for the table

mean_vector = np.zeros((len(subj_list), len(epoch_list)))
std_vector = np.zeros((len(subj_list), len(epoch_list)))

for i in range(len(subj_list)):
    subj = subj_list[i]
    for j in range(len(epoch_list)):
        epoch = epoch_list[j]
        
        mean_vector[i, j] = recon_loss_average_mean[subj][j]
        std_vector[i, j] = recon_loss_average_std[subj][j]