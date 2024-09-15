"""
Visualize the results obtained with script 10
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj = 3

epoch = 80
use_test_set = False
repetition = 5

# How many the time use the std to scale/shift the signal
n_change_start = 0
n_change_end = 9
n_change_step = 1

use_dtw_divergence = True

batch_size = 72
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if use_test_set : string_dataset = 'test'
else : string_dataset = 'train'

# Load error matrices
path_save = 'Saved Results/d2a_analysis/shift_and_scale_error/S{}/{}/'.format(subj, string_dataset)
path_full_matrix = path_save + 'full_matrix_epoch_{}_rep_{}_std_{}_{}_{}.npy'.format(epoch, repetition, n_change_start, n_change_end, n_change_step)
full_matrix_error = np.load(path_full_matrix)
path_avg_matrix = path_save + 'avg_matrix_epoch_{}_rep_{}_std_{}_{}_{}.npy'.format(epoch, repetition, n_change_start, n_change_end, n_change_step)
avg_matrix_error = np.load(path_avg_matrix)

# Normalize avg_matrix_error
avg_matrix_error_norm = avg_matrix_error / avg_matrix_error[0, 0]
