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
epoch = 80
subj_for_weights = 2
subj_for_data = 2
use_test_set = True

t_min = 2
t_max = 4

compute_spectra_with_entire_signal = True
nperseg = 500

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = False
plot_to_create = 20

repetition = 7
n_trial = 49 
channel = 'C6'
# channel = np.random.choice(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
#        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
#        'P2', 'POz'])
    

plot_config = dict(
    use_TkAgg_backend = True,
    rescale_minmax = False,
    figsize_time = (10, 5),
    figsize_freq = (10, 5),
    fontsize = 18,
    linewidth_original = 1.5,
    linewidth_reconstructed = 1.5,
    color_original = 'black',
    color_reconstructed = 'red',
    add_title = False,
    save_fig = True,
    format_so_save = ['png', 'pdf', 'eps']
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

xticks_time = None
# xticks_time = [2, 3, 4, 5, 6]

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}
if rand_trial_sample is False: plot_to_create = 1

dataset_config = cd.get_moabb_dataset_config([subj_for_data])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

std_train = train_dataset.data.std(-1).mean()
# dataset.data *= 1
# dataset.data += 900

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

    # x += 1000
    # x *= (1 * std_train)
    # plt.hist(x.flatten(), bins = 100)
    # plt.show()

    tmp_t = np.linspace(2, 6, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    label_name = label_dict[int(label)]
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj_for_weights, repetition, epoch)
    # path_weight = 'Saved Model/test_SDTW_divergence/S{}/model_{}.pth'.format(subj,epoch) # TODO remember remove
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

    # (OPTIONAL) 
    if plot_config['rescale_minmax'] :
        x_original_to_plot = (x_original_to_plot - x_original_to_plot.min()) / (x_original_to_plot.max() - x_original_to_plot.min())
        x_r_to_plot = (x_r_to_plot - x_r_to_plot.min()) / (x_r_to_plot.max() - x_r_to_plot.min())
    
    fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize_time'])
    
    ax_time.plot(t, x_original_to_plot, label = 'original signal',
                 color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax_time.plot(t, x_r_to_plot, label = 'reconstructed signal',
                 color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'], alpha = 0.7)
    ax_time.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
    ax_time.set_ylabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])
    ax_time.legend()
    if xticks_time is not None: ax_time.set_xticks(xticks_time)
    ax_time.set_xlim([t_min, t_max])
    ax_time.grid(True)
    ax_time.tick_params(axis = 'both', labelsize = plot_config['fontsize'])
    
    if plot_config['add_title'] : 
        if subj_for_weights == subj_for_data :
            ax_time.set_title('S{} - Ch. {} - Trial {}'.format(subj_for_data, channel, n_trial))
        else :
            ax_time.set_title('S{} - Ch. {} - Trial {} - Weights of S{}'.format(subj_for_data, channel, n_trial, subj_for_weights))
    
    fig_time.tight_layout()
    fig_time.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in frequency domain

    if plot_config['rescale_minmax'] :
        x_psd = (x_psd - x_psd.min()) / (x_psd.max() - x_psd.min())
        x_r_psd = (x_r_psd - x_r_psd.min()) / (x_r_psd.max() - x_r_psd.min())

    fig_freq, ax_freq = plt.subplots(1, 1, figsize = plot_config['figsize_freq'])
    ax_freq.plot(f, x_psd, label = 'original signal',
                 color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax_freq.plot(f, x_r_psd, label = 'reconstructed signal',
                 color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])

    ax_freq.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
    ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$]", fontsize = plot_config['fontsize'])
    ax_freq.set_xlim([0, 80])
    # ax_freq.legend()
    ax_freq.grid(True)
    ax_freq.tick_params(axis = 'both', labelsize = plot_config['fontsize'])

    if plot_config['add_title']: 
        if subj_for_weights == subj_for_data :
            ax_freq.set_title('S{} - Ch. {} - Trial {}'.format(subj_for_weights, channel, n_trial))
        else :
            ax_freq.set_title('S{} - Ch. {} - Trial {} - Weights of S{}'.format(subj_for_data, channel, n_trial, subj_for_weights))

    fig_freq.tight_layout()
    fig_freq.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if plot_config['save_fig']:

        if subj_for_weights != subj_for_data :
            path_save = "Saved Results/d2a_analysis/transfer_learning/subj {}/".format(subj_for_data)
        else :
            path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/".format(tot_epoch_training, subj_for_weights)

        os.makedirs(path_save, exist_ok = True)
        path_save += "s{}_trial_{}_{}_rep_{}_epoch_{}".format(subj_for_data, n_trial + 1, channel, repetition, epoch)
        if plot_config['rescale_minmax'] : path_save += '_minmax'

        if use_test_set: path_save += '_test_set'
        else: path_save += '_train_set'

        if subj_for_weights != subj_for_data : path_save += '_weights_S{}'.format(subj_for_weights)

        for format in plot_config['format_so_save']:
            fig_time.savefig(path_save + "_time." + format, format = format)
            fig_freq.savefig(path_save + "_freq." + format, format = format)
