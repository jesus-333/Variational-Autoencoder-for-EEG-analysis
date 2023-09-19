"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute the average reconstruction error, std and variation coefficient
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import random

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj_list = [2, 9]
repetition_list = np.arange(19) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

recon_loss_results = dict()

for subj in subj_list:
    recon_loss_results[subj] = dict()
    for epoch in epoch_list:
        recon_loss_results[subj][epoch] = []
        for repetition in repetition_list:
            path_load = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, subj, epoch, repetition)
            recon_loss_results[subj][epoch].append(np.load(path_load))
            
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

n_training_to_average = [3, 5, 8, 14, 19]

mean_recon_error_per_n_training = dict()
std_recon_error_per_n_training = dict()
variation_coef_recon_error_per_n_training = dict()

stats_mean_recon_error = dict()
stats_std_recon_error = dict()
stats_variation_coe_recon_error = dict()

for subj in subj_list:
    mean_recon_error_per_n_training[subj] = dict()
    std_recon_error_per_n_training[subj] = dict()
    variation_coef_recon_error_per_n_training[subj] = dict()
    
    stats_mean_recon_error[subj] = dict()
    stats_std_recon_error[subj] = dict()
    stats_variation_coe_recon_error[subj] = dict()
    
    stats_mean_recon_error[subj]['min'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_mean_recon_error[subj]['avg'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_mean_recon_error[subj]['max'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_std_recon_error[subj]['min']  = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_std_recon_error[subj]['avg']  = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_std_recon_error[subj]['max']  = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_variation_coe_recon_error[subj]['min'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_variation_coe_recon_error[subj]['avg'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    stats_variation_coe_recon_error[subj]['max'] = np.zeros((len(n_training_to_average), len(epoch_list)))
    
    for i in range(len(epoch_list)):
        epoch = epoch_list[i]
        mean_recon_error_per_n_training[subj][epoch] = dict()
        std_recon_error_per_n_training[subj][epoch] = dict()
        variation_coef_recon_error_per_n_training[subj][epoch] = dict()
        for j in range(len(n_training_to_average)):
            n_training = n_training_to_average[j]
            mean_recon_error_per_n_training[subj][epoch][n_training] = dict()
            std_recon_error_per_n_training[subj][epoch][n_training] = dict()
            variation_coef_recon_error_per_n_training[subj][epoch][n_training] = dict()
            
            training_results = np.asanyarray(random.sample(recon_loss_results[subj][epoch], n_training))
            
            mean_recon_error_per_n_training[subj][epoch][n_training] = training_results.mean(0)
            std_recon_error_per_n_training[subj][epoch][n_training] = training_results.std(0)
            variation_coef_recon_error_per_n_training[subj][epoch][n_training] = std_recon_error_per_n_training[subj][epoch][n_training] / mean_recon_error_per_n_training[subj][epoch][n_training]
            
            print("Subj {}, Epoch {}, N. training {}".format(subj, epoch, n_training), variation_coef_recon_error_per_n_training[subj][epoch][n_training].mean())           
            
            stats_mean_recon_error[subj]['min'][j, i] =  mean_recon_error_per_n_training[subj][epoch][n_training].min()
            stats_mean_recon_error[subj]['avg'][j, i] =  mean_recon_error_per_n_training[subj][epoch][n_training].mean()
            stats_mean_recon_error[subj]['max'][j, i] =  mean_recon_error_per_n_training[subj][epoch][n_training].max()
            
            stats_std_recon_error[subj]['min'][j, i] =  std_recon_error_per_n_training[subj][epoch][n_training].min()
            stats_std_recon_error[subj]['avg'][j, i] =  std_recon_error_per_n_training[subj][epoch][n_training].mean()
            stats_std_recon_error[subj]['max'][j, i] =  std_recon_error_per_n_training[subj][epoch][n_training].max()
            
            stats_variation_coe_recon_error[subj]['min'][j, i] =  variation_coef_recon_error_per_n_training[subj][epoch][n_training].min()
            stats_variation_coe_recon_error[subj]['avg'][j, i] =  variation_coef_recon_error_per_n_training[subj][epoch][n_training].mean()
            stats_variation_coe_recon_error[subj]['max'][j, i] =  variation_coef_recon_error_per_n_training[subj][epoch][n_training].max()
