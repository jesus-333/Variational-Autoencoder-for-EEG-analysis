"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial in different plot
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import os

import toml
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.dataset import dataset_time as ds_time
from library.training import train_generic
from library.analysis import support

from library import check_config

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# N.b. Per ora il percorso dei pesi è hardcoded
filename = 'NO_NOTCH_train8'
tot_epoch_training = 10
epoch = 10
use_test_set = False

t_min = 1
t_max = 5

compute_spectra_with_entire_signal = True
nperseg = 500

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = False
plot_to_create = 10

n_trial = 33
channel = 12
# channel = np.random(['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])

plot_config = dict(
    figsize_time = (10, 5),
    figsize_freq = (10, 5),
    fontsize = 18,
    linewidth_original = 1.5,
    linewidth_reconstructed = 1.5,
    color_original = 'black',
    color_reconstructed = 'red',
    add_title = False,
    save_fig = False,
    # format_so_save = ['png', 'pdf', 'eps']
    format_so_save = ['png']
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

xticks_time = None
# xticks_time = [2, 3, 4, 5, 6]

path_dataset_config = 'training_scripts/config/TUAR/dataset.toml'
path_model_config = 'training_scripts/config/TUAR/model.toml'
path_traing_config = 'training_scripts/config/TUAR/training.toml'


#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset and model creation

if rand_trial_sample == False : plot_to_create = 1
plt.rcParams.update({'font.size': plot_config['fontsize']})


# Load data
data_train = np.load('data/TUAR/NO_NOTCH_train8.npz')['train_data']
data_test = np.load('data/TUAR/NO_NOTCH_train8.npz')['test_data']

# Create train and validation dataset
dataset_train = ds_time.EEG_Dataset(data_train, np.ones(len(data_train)), ch_list = [])
dataset_test = ds_time.EEG_Dataset(data_test, np.ones(len(data_test)), ch_list = [])

# Crate fake labels array
labels_train = np.ones(len(data_train))
labels_test = np.ones(len(data_test))

# Get number of channels and length of time samples
C = data_train.shape[2]
T = data_train.shape[3]

# Model creation
model_config = toml.load(path_model_config)
check_config.check_model_config_hvEEGNet(model_config)
model_config['encoder_config']['C'] = C
model_config['encoder_config']['T'] = T
model_config['encoder_config']['c_kernel_2'] = [C, 1]
model = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)

# Decide if use the train or the test dataset
if use_test_set : dataset = data_test
else : dataset = data_train

for n_plot in range(plot_to_create):

    np.random.seed(None)
    if rand_trial_sample:
        n_trial = np.random.randint(len(dataset))
        # repetition = np.random.randint(19) + 1
        channel = np.random.choice(['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])

        # Da usare solo per il test fatto per sbaglio con 12 canali
        channel = np.random.choice(['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2']) 
        dataset.ch_list = np.asarray(['FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])
    
    # Get trial and create vector for time and channel
    x, label = dataset[n_trial]
    tmp_t = np.linspace(1, 5, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/TUAR/{}_{}_epochs/model_{}.pth'.format(filename, tot_epoch_training, epoch) 
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

    if plot_config['add_title']: ax_time.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))
    
    fig_time.tight_layout()
    fig_time.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in frequency domain

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

    if plot_config['add_title']: ax_freq.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))

    fig_freq.tight_layout()
    fig_freq.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if plot_config['save_fig']:
        path_save = "Saved Results/zhou2016/reconstruction/subj {}/".format(subj)

        if use_test_set: path_save += '/test/'
        else: path_save += '/train/'

        os.makedirs(path_save, exist_ok = True)
        # path_save += "subj_{}_trial_{}_ch_{}_rep_{}_epoch_{}_label_{}".format(subj, n_trial + 1, channel, repetition, epoch, label_name)
        path_save += "s{}_trial_{}_{}_rep_{}_epoch_{}".format(subj, n_trial + 1, channel, repetition, epoch)

        if use_test_set: path_save += '_test_set'
        else: path_save += '_train_set'

        for format in plot_config['format_so_save']:
            fig_time.savefig(path_save + "_time." + format, format = format)
            fig_freq.savefig(path_save + "_freq." + format, format = format)
