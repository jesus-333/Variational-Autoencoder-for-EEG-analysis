"""
Visualize the results obtained with script 11
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
n_list = np.arange(16)

plot_config = dict(
    use_TkAgg_backend = False,
    figsize = (12, 8),
    fontsize = 16,
    use_grid = True,
    title = 'Reconstruction error vs shift',
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend'] : plt.switch_backend('TkAgg')

if use_test_set : string_dataset = 'test'
else : string_dataset = 'train'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data

# Variable to save results
avg_error = np.zeros(len(n_list))


# Load the error matrix for each value of n
for i in range(len(n_list)) : 
    n = n_list[i]

    # Load error matrix
    path_save = 'Saved Results/d2a_analysis/shift_error/S{}/{}/'.format(subj, string_dataset)
    path_recon_matrix = path_save + 'recon_matrix_epoch_{}_rep_{}_std_{}.npy'.format(epoch, repetition, n)
    recon_matrix_error = np.load(path_recon_matrix)
    
    # Compute and save average error
    avg_error[i] = recon_matrix_error.mean()

# Normalize the average error
# This make sense only if the first element of the list contains the error computed without shift
# In this case, all others errors represents the error respect the case without shift
avg_error_norm = avg_error / avg_error[0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the results

# Create the plot
fig, ax = plt.subplots(figsize = plot_config['figsize'])

# Plot the results
ax.plot(n_list, avg_error_norm, marker = 'o', linestyle = '--', color = 'b', label = 'Reconstruction error')

# Other plot settings
ax.set_xlabel('Shift (n times train std)', fontsize = plot_config['fontsize'])
ax.set_ylabel('Reconstruction error S{}'.format(subj), fontsize = plot_config['fontsize'])
if plot_config['use_grid'] : ax.grid(True)
if 'title' in plot_config : ax.set_title(plot_config['title'], fontsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()
