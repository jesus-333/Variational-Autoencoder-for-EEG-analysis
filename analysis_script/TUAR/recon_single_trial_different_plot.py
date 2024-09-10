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

def low_pass_data(data, config : dict) :
    b, a = signal.butter(config['order'], Wn = config['cutoff'], fs = config['fs'], btype='lowpass')
    return signal.filtfilt(b, a, data)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# N.b. RICORDATI NORMALIZATION DEI TRAILS
filename = 'NO_NOTCH_train8'
filename = 'NO_NOTCH_shuffle6'
tot_epoch_training = 30
epoch = 30
use_test_set = True

t_min = 0
t_max = 4

compute_spectra_with_entire_signal = True
nperseg = None

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = True
plot_to_create = 20

n_trial = 327
channel = 10
# channel = np.random(['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])

filter_config = dict(
    use_filter = False,
    fs = 250,
    cutoff = 80,
    order = 5
)

plot_config = dict(
    rescale_minmax = True,
    figsize_time = (15, 8),
    figsize_freq = (15, 8),
    fontsize = 18,
    linewidth_original = 1.5,
    linewidth_reconstructed = 1.5,
    color_original = 'black',
    color_reconstructed = 'red',
    add_title = False,
    save_fig = True,
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
data_train = np.load('data/TUAR/{}.npz'.format(filename))['train_data']
data_test = np.load('data/TUAR/{}.npz'.format(filename))['test_data']

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
model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
# model_hv.eval()

# Decide if use the train or the test dataset
if use_test_set : dataset = dataset_test
else : dataset = dataset_train

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot

for n_plot in range(plot_to_create):

    np.random.seed(None)
    if rand_trial_sample:
        n_trial = np.random.randint(len(dataset))
        channel = np.random.randint(data_train.shape[2])
    
    # Get trial and create vector for time and channel
    x, _ = dataset[n_trial]
    tmp_t = np.linspace(0, 4, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = np.arange(data_train.shape[2]) == channel
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/TUAR/{}_{}_epochs/model_{}.pth'.format(filename, tot_epoch_training, epoch) 
    # path_weight = 'Saved Model/repetition_hvEEGNet_80/subj 3/rep 13/model_80.pth'
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r = model_hv.reconstruct(x.unsqueeze(0)).squeeze()

    if filter_config['use_filter']:
        x_r = low_pass_data(x_r[idx_ch], filter_config).squeeze()
    else : 
        x_r = x_r[idx_ch].squeeze()
    
    # Select channel and time samples
    x_original_to_plot = x[0, idx_ch, idx_t]
    x_r_to_plot = x_r[idx_t]
    
    if compute_spectra_with_entire_signal:
        x_original_for_psd = x[0, idx_ch, :].squeeze()
        x_r_for_psd = x_r[:].squeeze()
    else:
        x_original_for_psd = x_original_to_plot
        x_r_for_psd = x_r_to_plot

    
    # Compute PSD
    fs = 250
    f, x_psd = signal.welch(x_original_for_psd, fs = fs, nperseg = nperseg)
    f, x_r_psd = signal.welch(x_r_for_psd, fs = fs, nperseg = nperseg)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in time domain
    
    fig_time, ax_time = plt.subplots(1, 1, figsize = plot_config['figsize_time'])
    
    # TMP
    if plot_config['rescale_minmax'] :
        x_original_to_plot = (x_original_to_plot - x_original_to_plot.min()) / (x_original_to_plot.max() - x_original_to_plot.min())
        x_r_to_plot = (x_r_to_plot - x_r_to_plot.min()) / (x_r_to_plot.max() - x_r_to_plot.min())
    
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

    if plot_config['add_title']: ax_time.set_title('{} - Ch. {} - Trial {}'.format(filename, channel, n_trial))
    
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
    ax_freq.set_xlim([0, fs/2])
    ax_freq.legend()
    ax_freq.grid(True)
    ax_freq.tick_params(axis = 'both', labelsize = plot_config['fontsize'])

    if plot_config['add_title']: ax_freq.set_title('{} - Ch. {} - Trial {}'.format(filename, channel, n_trial))

    fig_freq.tight_layout()
    fig_freq.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if plot_config['save_fig']:
        path_save = "Saved Results/TUAR/reconstruction/{}/".format(filename)

        if use_test_set : path_save += '/test/'
        else: path_save += '/train/'

        os.makedirs(path_save, exist_ok = True)
        # path_save += "subj_{}_trial_{}_ch_{}_rep_{}_epoch_{}_label_{}".format(subj, n_trial + 1, channel, repetition, epoch, label_name)
        path_save += "{}_trial_{}_{}_epoch_{}".format(filename, n_trial + 1, channel, epoch)

        if filter_config['use_filter'] : path_save += 'low_pass_filter'

        if use_test_set: path_save += '_test_set'
        else: path_save += '_train_set'

        for format in plot_config['format_so_save']:
            fig_time.savefig(path_save + "_time." + format, format = format)
            fig_freq.savefig(path_save + "_freq." + format, format = format)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if rand_trial_sample : plt.close('all')


