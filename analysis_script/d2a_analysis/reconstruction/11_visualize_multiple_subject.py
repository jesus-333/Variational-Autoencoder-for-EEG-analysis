"""
Visualize the results obtained with script 11 with multiple subject on a single plot
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [1, 2, 3, 4]
repetition_list = [1, 2, 5]

epoch = 80
use_test_set = False

# How many the time use the std to scale/shift the signal
n_list = np.arange(16)

plot_config = dict(
    use_TkAgg_backend = False,
    add_std = False,
    method_to_compute_avg = 1,
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
avg_error = np.zeros((len(subj_list), len(repetition_list), len(n_list)))
avg_error_norm = np.zeros((len(subj_list), len(repetition_list), len(n_list)))

# Load error matrix for each subject and values of n
for i in range(len(subj_list)) :
    subj = subj_list[i]

    for j in range(len(repetition_list)) :
        repetition = repetition_list[j]

        for k in range(len(n_list)) : 
            n = n_list[k]

            # Load error matrix
            path_save = 'Saved Results/d2a_analysis/shift_error/S{}/{}/'.format(subj, string_dataset)
            path_recon_matrix = path_save + 'recon_matrix_epoch_{}_rep_{}_std_{}.npy'.format(epoch, repetition, n)
            recon_matrix_error = np.load(path_recon_matrix)
            
            # Compute and save average error
            avg_error[i, j, k] = recon_matrix_error.mean()

        # Normalize the average error
        # This make sense only if the first element of the list contains the error computed without shift
        # In this case, all others errors represents the error respect the case without shift
        avg_error_norm[i, j] = avg_error[i, j] / avg_error[i, j, 0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the results

# Create the plot
fig, ax = plt.subplots(figsize = plot_config['figsize'])

for i in range(len(subj_list)) :
    # Average along the repetition
    if plot_config['method_to_compute_avg'] == 1 :
        avg_to_plot = avg_error_norm[i].mean(0)
        # N.b. : avg_error_norm[i] has shape (len(repetition_list), len(n_list)) and I want to the mean over the repetition
    elif plot_config['method_to_compute_avg'] == 2 : 
        avg_to_plot = avg_error[i].mean(0) / avg_error[i].mean(0)[0]
    
    # Plot the results
    ax.plot(n_list, avg_to_plot,
            marker = 'o', linestyle = '--', 
            label = 'S{}'.format(subj_list[i])
            )

    if plot_config['add_std'] and subj_list[i] != 3:
        if plot_config['method_to_compute_avg'] == 1 :
            std_to_use = avg_error_norm[i].std(0)
        elif plot_config['method_to_compute_avg'] == 2 : 
            std_to_use = avg_error[i].std(0) / avg_error[i].std(0)[0]

        ax.fill_between(n_list, 
                        avg_to_plot - std_to_use,
                        avg_to_plot + std_to_use,
                        alpha = 0.2
                        )

# Other plot settings
ax.legend()
ax.set_xlabel('Shift (n times train std)', fontsize = plot_config['fontsize'])
ax.set_ylabel('Reconstruction error S{}'.format(subj), fontsize = plot_config['fontsize'])
if plot_config['use_grid'] : ax.grid(True)
if 'title' in plot_config : ax.set_title(plot_config['title'], fontsize = plot_config['fontsize'])
# ax.set_ylim([0, 100])

fig.tight_layout()
fig.show()
