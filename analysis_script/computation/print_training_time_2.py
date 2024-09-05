"""
Print the results obtained with the script training_time_2.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path = 'Saved Results/computation time/'
machine_to_plot = 'CUDA_Jetson_nano'

n_elements_dataset_list = [25, 50, 75, 100, 125, 150, 175, 200]

batch_size_list = [2, 3, 5, 7, 8, 10, 12, 13, 15]

plot_config = dict(
    figsize = (15, 10),
    cmap = 'RdYlGn_r',
    # cmap = None,
    interpolation = 'bilinear',
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

path = path + machine_to_plot + '/'

average_training_time_matrix = np.zeros((len(batch_size_list), len(n_elements_dataset_list)))

for i in range(len(batch_size_list)) :
    batch_size = batch_size_list[i]
    for j in range(len(n_elements_dataset_list)) :
        n_elements_dataset = n_elements_dataset_list[j]
        
        # Load file
        path_file = path + 'n_elements_{}_batch_{}/time_list.npy'.format(n_elements_dataset, batch_size)
        time_list = np.load(path_file)

        # Compute average training time
        average_training_time_matrix[i, j] = np.mean(time_list)

        if np.mean(time_list) > 100 :
            print(n_elements_dataset, batch_size, np.mean(time_list))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot data

# Create figure
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Plot data
im = ax.imshow(average_training_time_matrix,
               cmap = plot_config['cmap'], interpolation = plot_config['interpolation'],
               )

# Add other information
ax.set_xlabel('Number of elements in the dataset')
ax.set_ylabel('Batch size')
ax.set_xticks([-0.5], labels = [])
ax.set_yticks([-0.5], labels = [])
ax.set_xticks(np.arange(len(n_elements_dataset_list)) + 0.5, labels = n_elements_dataset_list, minor = True)
ax.set_yticks(np.arange(len(batch_size_list)) + 0.5, labels = batch_size_list, minor = True)
# ax.grid(True)

# Add colorbar
cbar = plt.colorbar(im, ax = ax)

# Show Figure
fig.tight_layout()
fig.show()

if plot_config['save_fig'] :
    # Save figure
    path_save = path + 'average_training_time'
    fig.savefig(path_save + '.png', format = 'png')
    fig.savefig(path_save + '.pdf', format = 'pdf')
