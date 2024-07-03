"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial in the same plot
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.analysis import support
from library.config import config_dataset as cd

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

subj = 3
use_test_set = False

t_min = 2
t_max = 6

nperseg = 500

n_trial_1 = 0
channel_1 = 'C4'

n_trial_2 = 9
channel_2 = 'Fz'

plot_config = dict(
    figsize = (18, 12),
    fontsize = 15,
    t_limit = [2, 4],
    freq_limit = [0, 50],
    color = 'black' ,
    add_subtitle = True,
    save_fig = True,
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
label_dict = {0 : 'left hand', 1 : 'right hand', 2 : 'foot', 3 : 'tongue'}

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , _ = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

# Get trial and create vector for time and channel
x, label_1 = dataset[n_trial_1]
x_first = x.squeeze()[dataset.ch_list == channel_1].squeeze()

# Select channel and time samples
x, label_2 = dataset[n_trial_2]
x_second = x.squeeze()[dataset.ch_list == channel_2].squeeze()

t = np.linspace(2, 6, x_first.shape[-1])

# Compute PSD
f, x_first_psd = signal.welch(x_first, fs = 250, nperseg = nperseg)
f, x_second_psd = signal.welch(x_second, fs = 250, nperseg = nperseg)

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, ax = plt.subplots(2, 2, figsize = plot_config['figsize'])

ax[0, 0].plot(t, x_first, label = 'Reconstruct Signal', color = 'black')
ax[0, 1].plot(f, x_first_psd, label = 'Reconstruct Signal', color = 'black')

ax[1, 0].plot(t, x_second, label = 'Reconstruct Signal', color = 'black')
ax[1, 1].plot(f, x_second_psd, label = 'Reconstruct Signal', color = 'black')

for i in range(2):
    for j in range(2):
        ax[i, j].grid(True)
        ax[i, j].tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
        ax[i, j].tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'])
        
        if j == 0 : # Time domain
            ax[i, j].set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
            if 't_limit' in plot_config : ax[i, j].set_xlim([plot_config['t_limit'][0], plot_config['t_limit'][-1]])
            else : ax[i, j].set_xlim([t[0], t[-1]])
            ax[i, j].set_ylim([-30, 47])
            ax[i, j].set_ylabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])
        elif j == 1 : # Frequency domain
            ax[i, j].set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
            if 'freq_limit' in plot_config : ax[i, j].set_xlim([plot_config['freq_limit'][0], plot_config['freq_limit'][-1]])
            else : ax[i, j].set_xlim([f[0], f[-1]])
            ax[i, j].set_ylim([-1, 14])
            ax[i, j].set_ylabel(r"PSD [$\mu V^2/Hz$]", fontsize = plot_config['fontsize'])

if plot_config['add_subtitle'] :
    label_list = ["Segment no. {}, Ch. {}, {} movement, time domain".format(n_trial_1 + 1, channel_1, label_dict[int(label_1)]),
                  "Segment no. {}, Ch. {}, {} movement, frequency domain".format(n_trial_1 + 1, channel_1, label_dict[int(label_1)]),
                  "Segment no. {}, Ch. {}, {} movement, time domain".format(n_trial_2 + 1, channel_2, label_dict[int(label_2)]),
                  "Segment no. {}, Ch. {}, {} movement, frequency domain".format(n_trial_2 + 1, channel_2, label_dict[int(label_2)])
                  ]

    ax = ax.flatten()
    for i in range(len(ax)) :
        ax[i].text(0.5, -0.2, label_list[i], transform = ax[i].transAxes, ha = "center", fontsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/figure_paper_frontiers/"
    os.makedirs(path_save, exist_ok = True)
    path_save += "FIG_3_example_d2a"
    if plot_config['add_subtitle'] : path_save += "_with_subtitle"
    else : path_save += "_NO_subtitle"
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".jpeg", format = 'jpeg')
    fig.savefig(path_save + ".eps", format = 'eps')
