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

from library.dataset import download, dataset_time
from library.model import hvEEGNet

from library import check_config

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# N.b. Per ora il percorso dei pesi è hardcoded
tot_epoch_training = 20
epoch = 10
subj = 6 
repetition = 1
trained_with_test_data = False
use_test_set_for_reconstruction = False

t_min = 0
t_max = 1.5

compute_spectra_with_entire_signal = True
nperseg = 512

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = True
plot_to_create = 2

n_trial = 252
channel = 'F3'
# channel = np.random(['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])

plot_config = dict(
    figsize_time = (16, 8),
    figsize_freq = (16, 8),
    rescale_minmax = False,
    fontsize = 18,
    linewidth_original = 1.5,
    linewidth_reconstructed = 1.5,
    color_original = 'black',
    color_reconstructed = 'red',
    add_title = True,
    save_fig = True,
    # format_so_save = ['png', 'pdf', 'eps']
    format_so_save = ['png']
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

xticks_time = None
# xticks_time = [2, 3, 4, 5, 6]

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'rest', 1 : 'right_elbow_extension', 2 : 'right_elbow_flexion', 3 : 'right_hand_close', 4 : 'right_hand_open', 5 : 'right_pronation', 6 : 'right_supination'} # TODO CHECK

# Get subject data and model
if 'dataset_config' in locals():
    if subj != dataset_config['subjects_list'][0] :
        load_data = True
    else :
        load_data = False
else :
    load_data = True
print("Load data : ", load_data)
    
if load_data:
    dataset_config = toml.load('training_scripts/config/Ofner2017/dataset.toml')
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    dataset_config['subjects_list'] = [subj] 
    
    train_data, train_labels, ch_list = download.get_Ofner2017(dataset_config, 'train')
    test_data, test_labels, ch_list = download.get_Ofner2017(dataset_config, 'test')

# Add extra dimension, necessary to work with Conv2d layer
train_data = np.expand_dims(train_data, 1)
test_data = np.expand_dims(test_data, 1)

# # For some reason the total number of samples is 1251 instead of 1250 (if no resample is used)
# (The original signal is sampled at 250Hz for 5 seconds)
# In this case to have a even number of samples the last one is removed
# Note also that the final signal has length 1000 samples (4s) because the activity is performed from 1s to 5s
# You can change this by chaning the key trial_start and trials_end in dataset_time
if dataset_config['resample_data'] == False :
    train_data = train_data[:, :, :, 0:-1]
    test_data = test_data[:, :, :, 0:-1]

model_config = toml.load('training_scripts/config/Ofner2017/model.toml')
model_config['encoder_config']['C'] = train_data.shape[2]
model_config['encoder_config']['T'] = train_data.shape[3] 
check_config. check_model_config_hvEEGNet(model_config)
model_hv = hvEEGNet.hvEEGNet_shallow(model_config)

# Decide if use the train or the test dataset
if use_test_set_for_reconstruction  : 
    dataset = dataset_time.EEG_Dataset(test_data, test_labels, ch_list)
else : 
    dataset = dataset_time.EEG_Dataset(train_data, train_labels, ch_list)

for n_plot in range(plot_to_create):

    np.random.seed(None)
    if rand_trial_sample:
        n_trial = np.random.randint(len(dataset))
        # repetition = np.random.randint(19) + 1
        channel = np.random.choice(ch_list)
    
    # Get trial and create vector for time and channel
    x, label = dataset[n_trial]
    tmp_t = np.linspace(0, 3, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    label_name = label_dict[int(label)]
    
    # Load weight and reconstruction
    if trained_with_test_data : path_weights = 'Saved Model/Ofner2017/trained_with_TEST_data/'
    else : path_weights = 'Saved Model/Ofner2017/train_with_TRAIN_data/'
    path_weights += 'S{}_{}_epochs_rep_{}/model_{}.pth'.format(subj, tot_epoch_training, repetition, epoch) # TODO Eventulmente da modificare in futuro
    model_hv.load_state_dict(torch.load(path_weights, map_location = torch.device('cpu')))
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
    f, x_psd = signal.welch(x_original_for_psd, fs = 512, nperseg = nperseg)
    f, x_r_psd = signal.welch(x_r_for_psd, fs = 512, nperseg = nperseg)
    
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

    if plot_config['add_title']: ax_time.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))
    
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

    if plot_config['add_title']: ax_freq.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))

    fig_freq.tight_layout()
    fig_freq.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if plot_config['save_fig']:
        path_save = "Saved Results/Ofner2017/reconstruction/subj {}/".format(subj)

        if use_test_set_for_reconstruction : path_save += '/test/'
        else: path_save += '/train/'

        os.makedirs(path_save, exist_ok = True)
        # path_save += "subj_{}_trial_{}_ch_{}_rep_{}_epoch_{}_label_{}".format(subj, n_trial + 1, channel, repetition, epoch, label_name)
        path_save += "s{}_trial_{}_{}_rep_{}_epoch_{}".format(subj, n_trial + 1, channel, repetition, epoch)
        
        if use_test_set_for_reconstruction : path_save += '_test_set'
        else: path_save += '_train_set'
        
        if plot_config['rescale_minmax'] : path_save += '_minmax'

        for format in plot_config['format_so_save']:
            fig_time.savefig(path_save + "_time." + format, format = format)
            fig_freq.savefig(path_save + "_freq." + format, format = format)
