"""
Analysis of the reconstruction of the various layer of the hvEEGNet
(i.e. reconstruct using only the deepest latent space, the deepest and the middle or all three)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

subj_list = [2]
loss = 'dtw'
epoch_list = [20, 40, 'BEST']
epoch_list = [40]
idx_trial = 47
idx_ch = 11

compute_psd = False 
use_test_set = True

config = dict(
    t_min = 2,
    t_max = 4,
    figsize = (20, 8),
    fontsize = 12,
)

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function

def get_dataset_and_model(subj_list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)
    device = 'cpu'

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Create model (hvEEGNet)
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model_config['use_classifier'] = False
    model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
    model_hv.to(device)

    return train_dataset, validation_dataset, test_dataset , model_hv

def crop_signal(x, idx_ch, t_start, t_end, t_min, t_max):
    """
    Select a single channel according to idx_ch and crop it according to t_min and t_max provided in config
    t_start, t_end = Initial and final second of the x signal
    t_min, t_max = min and max to keep of the original signal
    """
    t      = np.linspace(t_start, t_end, x.shape[-1])
    x_crop = x.squeeze()[idx_ch, np.logical_and(t >= t_min, t <= t_max)]
    t_plot = t[np.logical_and(t >= t_min, t <= t_max)]

    return x_crop, t_plot

def plot_signal(ax, horizontal_axis_value, x, horizontal_axis_value_r, x_r, compute_psd):
    ax.plot(horizontal_axis_value, x, label = 'Original signal')
    ax.plot(horizontal_axis_value_r, x_r, label = 'Reconstructed signal')
    if compute_psd: ax.set_xlabel("Frequency [Hz]")
    else: ax.set_xlabel("Time [s]")
    ax.legend()
    ax.grid(True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

for subj in subj_list:
    for epoch in epoch_list:
        # Get datasets and model
        train_dataset, validation_dataset, test_dataset , model_hv = get_dataset_and_model([subj])

        if use_test_set: dataset = test_dataset
        else: dataset = train_dataset
        
        # Load model weight
        # path_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, 3, epoch)
        path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(80, subj, 2, epoch)
        model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
        
        # Get EEG trial
        x      = dataset[idx_trial][0]
        label  = label_dict[int(dataset[idx_trial][1])]
        x_plot, horizontal_axis_value = crop_signal(x.squeeze(), idx_ch, 2, 6, config['t_min'], config['t_max'])

        latent_space_to_ignore = [False, False, False]
        x_r_1 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
        x_r_1_plot, horizontal_axis_value_r_1 = crop_signal(x_r_1, idx_ch, 2, 6, config['t_min'], config['t_max'])

        latent_space_to_ignore = [True, False, False]
        x_r_2 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
        x_r_2_plot, horizontal_axis_value_r_2 = crop_signal(x_r_2, idx_ch, 2, 6, config['t_min'], config['t_max'])

        latent_space_to_ignore = [True, True, False]
        x_r_3 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
        x_r_3_plot, horizontal_axis_value_r_3 = crop_signal(x_r_3, idx_ch, 2, 6, config['t_min'], config['t_max'])

        if compute_psd:
            nperseg = 500
            horizontal_axis_value, x_plot = signal.welch(x.squeeze()[idx_ch], fs = 250, nperseg = nperseg)
            horizontal_axis_value_r_1, x_r_1_plot = signal.welch(x_r_1.squeeze()[idx_ch], fs = 250, nperseg = nperseg)
            horizontal_axis_value_r_2, x_r_2_plot = signal.welch(x_r_2.squeeze()[idx_ch], fs = 250, nperseg = nperseg)
            horizontal_axis_value_r_3, x_r_3_plot = signal.welch(x_r_3.squeeze()[idx_ch], fs = 250, nperseg = nperseg)

        plt.rcParams.update({'font.size': config['fontsize']})
        fig, ax = plt.subplots(1, 3, figsize = config['figsize'])

        plot_signal(ax[0], horizontal_axis_value, x_plot, horizontal_axis_value_r_1, x_r_1_plot, compute_psd)
        ax[0].set_title("All latent space")

        plot_signal(ax[1], horizontal_axis_value, x_plot, horizontal_axis_value_r_2, x_r_2_plot, compute_psd)
        ax[1].set_title("Deep and middle latent space")

        plot_signal(ax[2], horizontal_axis_value, x_plot, horizontal_axis_value_r_3, x_r_3_plot, compute_psd)
        ax[2].set_title("Only deep latent space")
        
        fig.suptitle("{} - Subj {} - Trial {} - Label: {} - Ch: {} - Loss : {}".format("hvEEGNet_woclf", subj, idx_trial, label, dataset.ch_list[idx_ch], loss))
        fig.tight_layout()
        fig.show()

