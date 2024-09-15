"""
Convert the dataset to an image an plot them
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os

from library.analysis import support

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

dataset_to_use = 'd2a'

subj = 1

plot_config = dict(
    use_TkAgg_backend = False,
    figsize = (20, 10),
    cmap = 'Reds',
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def plot_data(data, title, plot_config : dict) :
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    ax.imshow(data, cmap = plot_config['cmap'], interpolation='nearest', aspect='auto')
    
    # Add colorbar
    # cbar = plt.colorbar(ax.imshow(data, cmap = plot_config['cmap']))

    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Channels')

    fig.tight_layout()
    fig.show()

    return fig, ax


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend'] :
    plt.switch_backend('TkAgg')

# Get dataset config
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset

# Load the dataset
if dataset_to_use == 'd2a' :
    train_dataset, _, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Get the array (shape n_trials x n_channels x n_times)
train_data = train_dataset.data.numpy().squeeze()
test_data = test_dataset.data.numpy().squeeze()

# Variable to use for the plot
train_data_to_plot = np.zeros((train_data.shape[1], train_data.shape[0] * train_data.shape[2]))
test_data_to_plot = np.zeros((test_data.shape[1], test_data.shape[0] * test_data.shape[2]))

# Convert datasets to images
for i in range(train_data.shape[0]) :
    train_data_to_plot[:, i * train_data.shape[2] : (i + 1) * train_data.shape[2]] = train_data[i, :, :]
    test_data_to_plot[:, i * test_data.shape[2] : (i + 1) * test_data.shape[2]] = test_data[i, :, :]

# Plot train data
fig_train, ax_train = plot_data(train_data_to_plot, 'Train data S{}'.format(subj), plot_config)

# Plot test data
fig_test, ax_test = plot_data(test_data_to_plot, 'Test data S{}'.format(subj), plot_config)

if plot_config['save_fig'] :
    path_save = 'Saved Results/{}/dataset_to_image/S{}_image_dataset'.format(dataset_to_use,subj)

    os.makedirs(path_save, exist_ok = True)

    fig_train.savefig(path_save + '_train.png')
    fig_test.savefig(path_save + '_test.png')
