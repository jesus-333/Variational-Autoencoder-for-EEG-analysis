"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial in different plot
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
rand_trial_sample = False
use_test_set = False

t_min = 2
t_max = 6
compute_spectra_with_entire_signal = True

nperseg = 500

plot_to_create = 80

# If rand_trial_sample == True they are selected randomly below
repetition = 19
n_trial = 250
channel = 'C1'
    
epoch = 80

plot_config = dict(
    figsize_time = (12, 8),
    figsize_freq = (12, 8),
    fontsize = 24   , 
    linewidth_original = 2,
    linewidth_reconstructed = 1,
    save_fig = True,
)

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

xticks_time = None
xticks_time = [2, 3, 4, 5, 6]

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
if rand_trial_sample is False: plot_to_create = 1

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

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
    label_name = label_dict[int(label)]
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r = model_hv.reconstruct(x.unsqueeze(0)).squeeze()
    
    # Select channel and time samples
    x_original_to_plot = x[0, idx_ch, idx_t]
    x_r_to_plot = x_r[idx_ch, idx_t]
    
    if compute_spectra_with_entire_signal:
        x_original_for_psd = x[0, idx_ch, :].squeeze()
        x_r_for_psd = x_r[idx_ch, :].squeeze()
    else:
        x_original_for_psd = x_original_to_plot
        x_r_for_psd = x_r_to_plot
    
    # Compute PSD
    f, x_psd = signal.welch(x_original_for_psd, fs = 250, nperseg = nperseg)
    f, x_r_psd = signal.welch(x_r_for_psd, fs = 250, nperseg = nperseg)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Plot in time domain
    
    fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize_time'])
    
    ax_time.plot(t, x_original_to_plot, label = 'original signal', color = 'grey', linewidth = plot_config['linewidth_original'])
    ax_time.plot(t, x_r_to_plot, label = 'reconstructed signal', color = 'black', linewidth = plot_config['linewidth_reconstructed'])
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel(r"Amplitude [$\mu$V]")
    ax_time.set_xlim([t_min, t_max])
    # ax_time.legend()
    ax_time.grid(True)
    if xticks_time is not None: ax_time.set_xticks(xticks_time)
    
    fig_time.tight_layout()
    fig_time.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
    # Plot in frequency domain

    fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize_freq'])
    ax_freq.plot(f, x_psd, label = 'original signal', 
                 color = 'grey', linewidth = plot_config['linewidth_original'])
    ax_freq.plot(f, x_r_psd, label = 'reconstructed signal', 
                 color = 'black', linewidth = plot_config['linewidth_reconstructed'])
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$]")
    ax_freq.set_xlim([0, 80])
    # ax_freq.legend()
    ax_freq.grid(True) 

    fig_freq.tight_layout()
    fig_freq.show()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/".format(tot_epoch_training, subj)
        os.makedirs(path_save, exist_ok = True)
        path_save += "subj_{}_trial_{}_ch_{}_rep_{}_epoch_{}_label_{}".format(subj, n_trial + 1, channel, repetition, epoch, label_name)
        
        if use_test_set: path_save += '_test_set'
        else: path_save += '_train_set'
        
        fig_time.savefig(path_save + "_time.png", format = 'png')
        fig_time.savefig(path_save + "_time.pdf", format = 'pdf')
        fig_freq.savefig(path_save + "_freq.png", format = 'png')
        fig_freq.savefig(path_save + "_freq.pdf", format = 'pdf')
