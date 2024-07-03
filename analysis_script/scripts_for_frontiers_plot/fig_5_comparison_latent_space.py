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
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

subj = 3
epoch = 'BEST'
repetition = 2

tot_epoch_training = 80

idx_trial = 0
channel = 'C3'

compute_psd = True
use_test_set = True

t_min = 2,
t_max = 3.002,

plot_config = dict(
    fontsize = 24,
    linewidth_original = 1.2,
    linewidth_reconstructed = 1.4,
    color_original = 'black',
    color_reconstructed = 'red',
    add_subtitle = True,
    save_fig = True,
)

if compute_psd :
    plot_config['figsize'] = [36, 8]
else :
    plot_config['figsize'] = [24, 24]

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

def plot_signal(ax, horizontal_axis_value, x, horizontal_axis_value_r, x_r, compute_psd, plot_config, add_legend = False):
    ax.plot(horizontal_axis_value, x, label = 'Original signal',
            color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax.plot(horizontal_axis_value_r, x_r, label = 'Reconstructed signal',
            color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])

    if compute_psd: ax.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
    else: ax.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])

    ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'] - 2)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'] - 2)
    if not compute_psd : ax.set_xticks([2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.5, 5])
    ax.set_xlim([horizontal_axis_value_r[0], horizontal_axis_value_r[-1]])

    if add_legend : ax.legend(fontsize = plot_config['fontsize'])
    if plot_config['add_subtitle'] :
        ax.text(0.5, -0.2, plot_config['subtitle'],
                transform = ax.transAxes, ha = "center", fontsize = plot_config['fontsize'] - 1)

    ax.grid(True)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

if use_test_set :
    dataset = test_dataset
    string_dataset = 'test'
else :
    dataset = train_dataset
    string_dataset = 'train'

# Load model weight
# path_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, 3, epoch)
path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

# Get EEG trial
x      = dataset[idx_trial][0]
idx_ch = dataset.ch_list == channel
label  = label_dict[int(dataset[idx_trial][1])]
x_plot, horizontal_axis_value = crop_signal(x.squeeze(), idx_ch, 2, 6, t_min, t_max)

latent_space_to_ignore = [False, False, False]
x_r_1 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
x_r_1_plot, horizontal_axis_value_r_1 = crop_signal(x_r_1, idx_ch, 2, 6, t_min, t_max)

latent_space_to_ignore = [True, False, False]
x_r_2 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
x_r_2_plot, horizontal_axis_value_r_2 = crop_signal(x_r_2, idx_ch, 2, 6, t_min, t_max)

latent_space_to_ignore = [True, True, False]
x_r_3 = model_hv.h_vae.reconstruct_ignoring_latent_spaces(x.unsqueeze(0), latent_space_to_ignore).squeeze()
x_r_3_plot, horizontal_axis_value_r_3 = crop_signal(x_r_3, idx_ch, 2, 6, t_min, t_max)

if compute_psd:
    nperseg = 500
    horizontal_axis_value, x_plot = signal.welch(x[0, idx_ch, :].squeeze(), fs = 250, nperseg = nperseg)
    horizontal_axis_value_r_1, x_r_1_plot = signal.welch(x_r_1[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
    horizontal_axis_value_r_2, x_r_2_plot = signal.welch(x_r_2[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
    horizontal_axis_value_r_3, x_r_3_plot = signal.welch(x_r_3[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
    string_domain = 'freq'

    fig, axs = plt.subplots(1, 3, figsize = plot_config['figsize'])
else:
    string_domain = 'time'

    fig, axs = plt.subplots(3, 1, figsize = plot_config['figsize'])

if compute_psd : plot_config['subtitle'] = r"f) Output from $z_3$"
else : plot_config['subtitle'] = r"c) Time domain reconstruction at the output of $z_3$ (including information from $z2$ and $z1$)"
plot_signal(axs[2], horizontal_axis_value, x_plot, horizontal_axis_value_r_1, x_r_1_plot, compute_psd, plot_config, add_legend = not compute_psd)
# ax.set_title("All latent space")
fig.tight_layout()
fig.show()
    
if compute_psd : plot_config['subtitle'] = r"e) Output from $z_2$"
else : plot_config['subtitle'] = r"b) Time domain reconstruction at the output of $z_2$ (including information from $z2$)"
plot_signal(axs[1], horizontal_axis_value, x_plot, horizontal_axis_value_r_2, x_r_2_plot, compute_psd, plot_config)
# ax.set_title("Deep and middle latent space")
fig.tight_layout()
fig.show()

if compute_psd : plot_config['subtitle'] = r"d) Output from $z_2$"
else : plot_config['subtitle'] = r"a) Time domain reconstruction at the output of $z_1$"
plot_signal(axs[0], horizontal_axis_value, x_plot, horizontal_axis_value_r_3, x_r_3_plot, compute_psd, plot_config)
# ax.set_title("Only deep latent space")
fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/figure_paper_frontiers/"
    os.makedirs(path_save, exist_ok = True)
    if compute_psd : path_save += "FIG_5_2_latent_space_recon_freq"
    else : path_save += "FIG_5_1_latent_space_recon_time"
    if plot_config['add_subtitle'] : path_save += "_with_subtitle"
    else : path_save += "_NO_subtitle"
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".jpeg", format = 'jpeg')
    fig.savefig(path_save + ".eps", format = 'eps')
