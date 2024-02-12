"""
Compute the average reconstruction for a list of subjcet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import sys
import os

from torch.utils.data import DataLoader

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
use_test_set = False

epoch = 80
tot_epoch_training = 80

filter_data = False
fmin = 0.5
fmax = 5

latent_space_to_ignore = [True, True, False] # Use only deep latent space
# latent_space_to_ignore = [True, False, False] # Use deep and middle latent space
latent_space_to_ignore = [False, False, False] # Use all 3 latent spaces

t_min = 2
t_max = 6
channel = 'P2'
compute_psd = False

plot_config = dict(
    figsize = (24, 15),
    fontsize = 20,
    add_std = False,
    add_original = True, # Only in the 3 x 3
    alpha = 0.33,
    save_fig = True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_to_color = ['red', 'blue', 'black', 'green', 'orange', 'violet', 'pink', 'brown', 'cyan']

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Compute the average reconstruction

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }

x_avg_orig_list = []
x_avg_r_list = []
x_std_list = []

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Reconstruction subj {}".format(subj))
    
    # Get dataset config
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    dataset_config['filter_data'] = filter_data
    dataset_config['fmin'] = fmin
    dataset_config['fmax'] = fmax

    # Load/create data and model
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')
    
    # Select train/test data and create dataloader
    if use_test_set: 
        dataset = test_dataset
        string_dataset = 'test'
    else: 
        dataset = train_dataset
        string_dataset = 'train'
    dataloader = DataLoader(dataset, batch_size = 72)

    # Load model weights and move to device
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, 2, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    model_hv.to(device)
    
    x_orig = None
    x_r = None

    for batch_data, batch_label in dataloader : 
        tmp_x = batch_data.to(device)

        tmp_x_r = model_hv.h_vae.reconstruct_ignoring_latent_spaces(tmp_x, latent_space_to_ignore).squeeze()

        if x_r is None :
            x_orig = tmp_x
            x_r = tmp_x_r
        else :
            x_orig = torch.cat((x_orig, tmp_x), 0)
            x_r = torch.cat((x_r, tmp_x_r), 0)

    x_avg_orig_list.append(x_orig.mean(0).cpu())
    x_avg_r_list.append(x_r.mean(0).cpu())
    x_std_list.append(x_r.std(0).cpu())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot data (a line for subject in the same plot)

idx_ch = dataset.ch_list == channel
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

for i in range(len(subj_list)):
    subj = subj_list[i]
    color = subj_to_color[i]

    x_avg_orig = x_avg_orig_list[i]
    x_avg_r = x_avg_r_list[i]
    x_std = x_std_list[i]

    if compute_psd:
        nperseg = 500
        horizontal_axis_value, x_avg_r_plot = signal.welch(x_avg_r[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
        string_domain = 'freq'
    else :
        x_avg_orig_plot, horizontal_axis_value = support.crop_signal(x_avg_orig, idx_ch, 2, 6, t_min, t_max)
        x_avg_r_plot, horizontal_axis_value = support.crop_signal(x_avg_r, idx_ch, 2, 6, t_min, t_max)
        x_std_plot, horizontal_axis_value = support.crop_signal(x_std, idx_ch, 2, 6, t_min, t_max)
        string_domain = 'time'

    ax.plot(horizontal_axis_value, x_avg_r_plot, color = color,
            label = 'S{}'.format(subj), 
            )

    
    if plot_config['add_std']:
        ax.fill_between(horizontal_axis_value, x_avg_r_plot + x_std_plot, x_avg_r_plot - x_std_plot, 
                        color = color, alpha = plot_config['alpha']
                        )

if string_domain == 'freq': 
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"PSD [$\mu V^2/Hz$]")
elif string_domain == 'time': 
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Amplitude [$\mu$V]")

ax.set_xlim([horizontal_axis_value[0], horizontal_axis_value[-1]])
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# (OPTIONAL) If there are 9 subjcets in the list create also the 3 x 3 plot
if len(subj_list) == 9:

    idx_ch = dataset.ch_list == channel
    fig, axs = plt.subplots(3, 3, figsize = plot_config['figsize'])

    k = 0
    for i in range(3):
        for j in range(3):

            subj = subj_list[k]
            color = subj_to_color[k]

            x_avg_orig = x_avg_orig_list[i]
            x_avg_r = x_avg_r_list[i]
            x_std = x_std_list[i]
            k += 1

            ax = axs[i, j]
            
            # (OPTIONAL) Convert in PSD
            if compute_psd:
                nperseg = 500
                horizontal_axis_value, x_avg_r_plot = signal.welch(x_avg_r[idx_ch].squeeze(), fs = 250, nperseg = nperseg)
                string_domain = 'freq'
            else :
                x_avg_orig_plot, horizontal_axis_value = support.crop_signal(x_avg_orig, idx_ch, 2, 6, t_min, t_max)
                x_avg_r_plot, horizontal_axis_value = support.crop_signal(x_avg_r, idx_ch, 2, 6, t_min, t_max)
                x_std_plot, horizontal_axis_value = support.crop_signal(x_std, idx_ch, 2, 6, t_min, t_max)
                string_domain = 'time'

            # (Optioanl) plot the average of the original signal
            if plot_config['add_original']:
                ax.plot(horizontal_axis_value, x_avg_orig_plot, color = 'black',
                        label = 'S{} - Orig'.format(subj), 
                        )
            
            # Plot the average of the reconstructed signal
            ax.plot(horizontal_axis_value, x_avg_r_plot, color = color,
                    label = 'S{}'.format(subj) if plot_config['add_original'] == False else 'S{} - Recon'.format(subj), 
                    )
            
            if plot_config['add_std']:
                ax.fill_between(horizontal_axis_value, x_avg_r_plot + x_std_plot, x_avg_r_plot - x_std_plot, 
                                color = color, alpha = plot_config['alpha']
                                )

            if string_domain == 'freq': 
                ax.set_xlabel("Frequency [Hz]")
                ax.set_ylabel(r"PSD [$\mu V^2/Hz$]")
            elif string_domain == 'time': 
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(r"Amplitude [$\mu$V]")

            ax.set_xlim([horizontal_axis_value[0], horizontal_axis_value[-1]])
            ax.legend()
            ax.grid(True)
    
    fig.suptitle("Channel {}".format(channel))
    fig.tight_layout()
    fig.show()
    
    # TODO complete save figure
    fig.savefig("Only_deep_{}.png".format(channel))
