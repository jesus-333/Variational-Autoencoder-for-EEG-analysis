"""
Created on Fri Sep  1 10:03:59 2023

@author: Alberto Zancanaro (jesus)
@organization: University of Padua

Load the data obtained with reconstruction_3.py and compute 
"""

#%%

import numpy as np
import pickle

#%%

tot_epoch_training = 20
subj_list = [2, 9]
epoch_list = [5, 10, 15, 20]

recon_loss_results = dict()

for subj in subj_list:
    recon_loss_results[subj] = dict()
    for epoch in epoch_list:
        recon_loss_results[subj][epoch] = dict()
        path_load = 'Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}.npy'.format(tot_epoch_training, subj, epoch)
        recon_loss_results[subj][epoch] = np.load(path_load)