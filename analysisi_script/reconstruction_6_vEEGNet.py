"""
Reconstruct in time (and frequency) using vEEGNet (not hierarchical) trained with dtw
Reconstruct a specific trial and channel for a specific subject
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

from library.dataset import preprocess as pp
from library.config import config_dataset as cd, config_model as cm
from library.training import train_generic

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

tot_epoch_training = 80
subj = 3
rand_trial_sample = False
use_test_set = True

t_min = 2
t_max = 6

nperseg = 500

plot_to_create = 0

# If rand_trial_sample == True they are selected randomly below
repetition = 10
n_trial = 0
channel = 'C3'
    
epoch = 20

plot_config = dict(
    figsize = (12, 8),
    fontsize = 14,
    save_fig = True,
)

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get dataset and model

# Get dataset
dataset_config = cd.get_moabb_dataset_config([subj])
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

# Get model
C = 22
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
hidden_space_dimension = 64
type_encoder = 0
type_decoder = 0
model_config = cm.get_config_vEEGNet(C, T, hidden_space_dimension, type_encoder, type_decoder)
model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
model = train_generic.get_untrained_model('EEGNet', model_config)

# Other stuff
plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
    
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get trial and create vector for time and channel
x, label = dataset[n_trial]
tmp_t = np.linspace(2, 6, x.shape[-1])
idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
t = tmp_t[idx_t]
idx_ch = dataset.ch_list == channel

# Load weight and reconstruction
path_weight = 'Saved Model/vEEGNet_dtw/{}/model_{}.pth'.format(subj, epoch)
model.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
x_r = model.reconstruct(x.unsqueeze(0)).squeeze()

# Select channel and time samples
x = x.squeeze()[idx_ch, idx_t]
x_r = x_r[idx_ch, idx_t]

# Compute PSD
f, x_psd = signal.welch(x, fs = 250, nperseg = nperseg)
f, x_r_psd = signal.welch(x_r, fs = 250, nperseg = nperseg)

fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Plot in time domain

ax_time.plot(t, x, label = 'Original Signal', color = 'grey')
ax_time.plot(t, x_r, label = 'Reconstruct Signal', color = 'black')
ax_time.set_xlabel("Time [s]")
ax_time.set_ylabel(r"Amplitude [$\mu$V]")
ax_time.legend()
ax_time.grid(True)

fig_time.tight_layout()
fig_time.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
# Plot in frequency domain

fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize'])
ax_freq.plot(f, x_psd, label = 'Original Signal', color = 'grey')
ax_freq.plot(f, x_r_psd, label = 'Reconstruct Signal', color = 'black')
ax_freq.set_xlabel("Frequency [Hz]")
ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$]")
ax_freq.legend()
ax_freq.grid(True) 

fig_freq.tight_layout()
fig_freq.show()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['save_fig']:
    path_save = "Saved Results/vEEGNet/subj {}/Plot/".format(tot_epoch_training, subj)
    os.makedirs(path_save, exist_ok = True)
    path_save += "trial_{}_ch_{}_rep_{}_epoch_{}".format(n_trial, channel, repetition, epoch)
    fig_time.savefig(path_save + "_time.png", format = 'png')
    fig_time.savefig(path_save + "_time.pdf", format = 'pdf')
    fig_freq.savefig(path_save + "_freq.png", format = 'png')
    fig_freq.savefig(path_save + "_freq", format = 'pdf')
