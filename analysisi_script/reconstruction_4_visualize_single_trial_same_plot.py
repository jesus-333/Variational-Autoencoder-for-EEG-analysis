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

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

tot_epoch_training = 80
subj = 9
rand_trial_sample = True
use_test_set = True

t_min = 2
t_max = 6

nperseg = 500

plot_to_create = 80

# If rand_trial_sample == True they are selected randomly below
repetition = 10
n_trial = 0
channel = 'C3'
    
first_epoch = 10
second_epoch = 60

plot_config = dict(
    figsize = (18, 12),
    fontsize = 12,
    save_fig = True,
)

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }

dataset_config = cd.get_moabb_dataset_config([subj])
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

for n_plot in range(plot_to_create):

    np.random.seed(None)
    if rand_trial_sample: 
        n_trial = np.random.randint(len(dataset))
        repetition = np.random.randint(19) + 1
        channel = np.random.choice(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
               'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
               'P2', 'POz'])
    
    # Get trial and create vector for time and channel
    x, label = dataset[n_trial]
    tmp_t = np.linspace(2, 6, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, first_epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r_first = model_hv.reconstruct(x.unsqueeze(0)).squeeze()
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, second_epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r_second = model_hv.reconstruct(x.unsqueeze(0)).squeeze()
    
    # Select channel and time samples
    x = x.squeeze()[idx_ch, idx_t]
    x_r_first = x_r_first[idx_ch, idx_t]
    x_r_second = x_r_second[idx_ch, idx_t]
    
    # Compute PSD
    f, x_psd = signal.welch(x, fs = 250, nperseg = nperseg)
    f, x_r_first_psd = signal.welch(x_r_first, fs = 250, nperseg = nperseg)
    f, x_r_second_psd = signal.welch(x_r_second, fs = 250, nperseg = nperseg)
    
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, ax = plt.subplots(2, 2, figsize = plot_config['figsize'])
    
    ax[0, 0].plot(t, x, label = 'Original Signal')
    ax[0, 0].plot(t, x_r_first, label = 'Reconstruct Signal')
    ax[0, 1].plot(f, x_psd, label = 'Original Signal')
    ax[0, 1].plot(f, x_r_first_psd, label = 'Reconstruct Signal')
    
    ax[1, 0].plot(t, x, label = 'Original Signal')
    ax[1, 0].plot(t, x_r_second, label = 'Reconstruct Signal')
    ax[1, 1].plot(f, x_psd, label = 'Original Signal')
    ax[1, 1].plot(f, x_r_second_psd, label = 'Reconstruct Signal')
    
    psd_max = max(x_psd.max(), x_r_first_psd.max(), x_r_second_psd.max())
    
    for i in range(2):
        for j in range(2):
            ax[i, j].legend()
            ax[i, j].grid(True)
            
            if j == 0: # Time domain
                ax[i, j].set_xlabel("Time [s]")
                ax[i, j].set_xlim([t[0], t[-1]])
                # ax[i, j].set_ylim([x.min()*1.1, x.max()*1.1])
                ax[i, j].set_ylim([-35, 35])
            elif j == 1:# Frequency domain
                ax[i, j].set_xlabel("Frequency [Hz]")
                ax[i, j].set_xlim([f[0], 80])
                # ax[i, j].set_ylim([0, psd_max*1.1])
                ax[i, j].set_ylim([0, 35])
        
            if i == 0: # First epoch selected
                ax[i, j].set_title("EPOCH {}".format(first_epoch))
            elif i == 1: # Second epoch selected
                ax[i, j].set_title("EPOCH {}".format(second_epoch))
    
    # ax.title()
    fig.suptitle("Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/".format(tot_epoch_training, subj)
        os.makedirs(path_save, exist_ok = True)
        path_save += "trial_{}_ch_{}_rep_{}".format(n_trial, channel, repetition)
        fig.savefig(path_save)
    
