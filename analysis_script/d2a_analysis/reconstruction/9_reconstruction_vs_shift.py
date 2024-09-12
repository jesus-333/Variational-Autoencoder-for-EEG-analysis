"""
Shift the average of the signal by n times the standard deviation of the training data and visualize the reconstruzion
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from library.analysis import support
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj = 2

epoch = 55
use_test_set = False
repetition = 5

t_min = 2
t_max = 4

n_trial = 252 
channel = 'Cz'

std_to_add_list = (np.arange(10) + 1) * 10
std_to_add_list = [10, 20]

plot_config = dict(
    use_TkAgg_backend = True,
    rescale_minmax = True,
    figsize = (12, 6),
    fontsize = 18,
    linewidth_original = 1.5,
    linewidth_reconstructed = 1.5,
    add_title = True,
    color_original = 'black',
    color_reconstructed = 'red',
    save_fig = True,
) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

# Get dataset
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, _, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

# Load weight
path_weight = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj, repetition, epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

# Variable used during the plot
tmp_t = np.linspace(2, 6, train_dataset.data.shape[-1])
idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
t = tmp_t[idx_t]
idx_ch = dataset.ch_list == channel

# Compute the standard deviation of the training data
tmp_data = train_dataset.data.squeeze()
std_train = tmp_data.std(-1).mean()
del tmp_data

for i in range(len(std_to_add_list)) :
    std_to_add = std_to_add_list[i]

    # Get trial and create vector for time and channel
    x, label = dataset[n_trial]
    label_name = label_dict[int(label)]

    # Add the standard deviation to the signal
    value_to_add = std_train * std_to_add
    x_to_reconstruct = x + value_to_add
    print(value_to_add)

    # Reconstruction
    x_r = model_hv.reconstruct(x_to_reconstruct.unsqueeze(0)).squeeze()

    # Select channel and time samples
    x_original_to_plot = x[0, idx_ch, idx_t]
    x_r_to_plot = x_r[idx_ch, idx_t]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Plot orignal and reconstructed signal

    # (OPTIONAL) Rescale the signal
    if plot_config['rescale_minmax'] :
        x_original_to_plot = (x_original_to_plot - x_original_to_plot.min()) / (x_original_to_plot.max() - x_original_to_plot.min())
        x_r_to_plot = (x_r_to_plot - x_r_to_plot.min()) / (x_r_to_plot.max() - x_r_to_plot.min())

    fig, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax_time.plot(t, x_original_to_plot, label = 'original signal',
                 color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax_time.plot(t, x_r_to_plot, label = 'reconstructed signal',
                 color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'], alpha = 0.7)
    ax_time.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
    ax_time.set_ylabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])
    ax_time.legend()
    ax_time.set_xlim([t_min, t_max])
    ax_time.grid(True)
    ax_time.tick_params(axis = 'both', labelsize = plot_config['fontsize'])

    if plot_config['add_title'] : ax_time.set_title('S{} - Ch. {} - Trial {} ({} std)'.format(subj, channel, n_trial, std_to_add))
    
    fig.tight_layout()
    fig.show()

    # (OPTIONAL) Save the figure
    if plot_config['save_fig'] :
        path_to_save = 'Saved Results/d2a_analysis/reconstruction_vs_shift/S{}/'.format(subj)
        os.makedirs(path_to_save, exist_ok = True)
        fig.savefig(path_to_save + 'Trial_{}_Ch_{}_std_{}.png'.format(n_trial, channel, std_to_add), format = 'png')
