"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial
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

from library.analysis import support
from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.training.soft_dtw_cuda import SoftDTW

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

tot_epoch_training = 80
subj = 2
rand_trial_sample = True
use_test_set = False

t_min = 2
t_max = 6

if rand_trial_sample:
    repetition = np.random.randint(19) + 1
    epoch = np.random.randint(16) * 5 # N.b. epoch are multiple of 5
    # N. trial defined below
    channel = np.random.randint(22)
else:
    repetition = 10
    epoch = 25
    n_trial = 0
    channel = 'C3'

plot_config = dict(
    figsize = (15, 10),
)

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def plot_figure(t, x, x_r, plot_config):

    if plot_config['use_psd']: 
        t, x_plot = signal.welch(x, fs = 250, nperseg = nperseg)
        t, x_r_plot = signal.welch(x_r, fs = 250, nperseg = nperseg)

    fig, ax = plt.subplots(1, 1, plot_config['figsize'])

    ax.plot(t, x_plot.squeeze()[idx_ch, idx_t], label = 'Original Signal')
    ax.plot(t, x_r_plot.squeeze()[idx_ch, idx_t], label = 'Reconstruct Signal')

    ax.legend(True)
    ax.grid(True)
    ax.set_xlabel("Time [s]")
    ax.set_xlim([t[0], t[-1]])

    fig.tigh_layout()
    fig.show()

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model([subj])
path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

if rand_trial_sample: n_trial = np.random.randint(len(dataset))

# Get trial and reconstruct it
x, label = dataset[n_trial]
x_r = model_hv.reconstruct(x.unsqueeze(0))

# Create vector for time and channel
tmp_t = np.linspace(2, 6, x.shape[-1])
idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
t = tmp_t[idx_t]
idx_ch = dataset.ch_list == channel

# Select channel and time samples
x = x.squeeze()[idx_ch, idx_t]
x_r = x_r.squeeze()[idx_ch, idx_t]
