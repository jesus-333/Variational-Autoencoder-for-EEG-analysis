"""
Analysis of the reconstruction of the various layer of the hvEEGNet
(i.e. reconstruct using only the deepest latent space, the deepest and the middle or all three)
"""
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

subj = 3
loss = 'dtw'
epoch_list = [20, 40, 'BEST']
epoch_list = [40]

tot_epoch_training = 80

idx_trial = 0
channel = 'C3'

compute_psd = False 
use_test_set = True

config = dict(
    t_min = 2,
    t_max = 3,
)

plot_config = dict(
    figsize = (24, 8),
    fontsize = 24,
    linewidth_original = 1,
    linewidth_reconstructed = 1,
    color_original = 'black',
    color_reconstructed = 'red',
    save_fig = True,
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function

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

def plot_signal(ax, horizontal_axis_value, x, horizontal_axis_value_r, x_r, compute_psd, plot_config):
    ax.plot(horizontal_axis_value, x, label = 'Original signal',
            color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax.plot(horizontal_axis_value_r, x_r, label = 'Reconstructed signal',
            color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])
    if compute_psd: ax.set_xlabel("Frequency [Hz]")
    else: ax.set_xlabel("Time [s]")
    ax.set_xlim([horizontal_axis_value_r[0], horizontal_axis_value_r[-1]])
    ax.legend()
    ax.grid(True)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

for epoch in epoch_list:

    if use_test_set: 
        dataset = test_dataset
        string_dataset = 'test'
    else: 
        dataset = train_dataset
        string_dataset = 'train'
    
    # Load model weight
    # path_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, 3, epoch)
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, 2, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    
    # Get EEG trial
    x      = dataset[idx_trial][0]
    idx_ch = dataset.ch_list == channel
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
        horizontal_axis_value, x_plot = signal.welch(x[0, idx_ch, :].squeeze(), fs = 250, nperseg = nperseg)
        horizontal_axis_value_r_1, x_r_1_plot = signal.welch(x_r_1[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
        horizontal_axis_value_r_2, x_r_2_plot = signal.welch(x_r_2[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
        horizontal_axis_value_r_3, x_r_3_plot = signal.welch(x_r_3[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
        string_domain = 'freq'
    else:
        string_domain = 'time'
    
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    plot_signal(ax, horizontal_axis_value, x_plot, horizontal_axis_value_r_1, x_r_1_plot, compute_psd, plot_config)
    # ax.set_title("All latent space")
    fig.tight_layout()
    fig.show()
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/{}/subj {}/Plot/".format(tot_epoch_training, string_dataset, subj)
        os.makedirs(path_save, exist_ok = True)
        path_save += "reconstruction_all_latent_space_{}".format(string_domain)
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')
        
    
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    plot_signal(ax, horizontal_axis_value, x_plot, horizontal_axis_value_r_2, x_r_2_plot, compute_psd, plot_config)
    # ax.set_title("Deep and middle latent space")
    fig.tight_layout()
    fig.show()
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/{}/subj {}/Plot/".format(tot_epoch_training, string_dataset, subj)
        os.makedirs(path_save, exist_ok = True)
        path_save += "reconstruction_deep_and_middle_latent_space_{}".format(string_domain)
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    plot_signal(ax, horizontal_axis_value, x_plot, horizontal_axis_value_r_3, x_r_3_plot, compute_psd, plot_config)
    # ax.set_title("Only deep latent space")
    fig.tight_layout()
    fig.show()
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/{}/subj {}/Plot/".format(tot_epoch_training, string_dataset, subj)
        os.makedirs(path_save, exist_ok = True)
        path_save += "reconstruction_only_deep_latent_space_{}".format(string_domain)
        fig.savefig(path_save + ".png", format = 'png')
        fig.savefig(path_save + ".pdf", format = 'pdf')

