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

from library.dataset import download, dataset_time, preprocess
from library.model import hvEEGNet

from library import check_config

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# N.b. Per ora il percorso dei pesi è hardcoded
epoch = 'BEST'
subj_id = 8
data_file_path = 'data/SEED/'
weight_file_path = 'Saved Model/SEED/subj_{}_bsdtw/'.format(subj_id)
use_test_set = False

trials_length_in_seconds = 4
t_min = 0
t_max = 3

compute_spectra_with_entire_signal = True
nperseg = 200

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = True
plot_to_create = 55

repetition = 5
n_trial = 252
channel = 41

plot_config = dict(
    figsize_time = (10, 5),
    figsize_freq = (10, 5),
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
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot'} # TODO CHECK


# Get all the file for a single subject
filename_list = []
list_files = os.listdir(data_file_path)
for file in list_files :
    if '_' in file :
        file_id = int(file.split('_')[0])
        if file_id == subj_id : filename_list.append(data_file_path + file)
    else :
        continue

# Get subject data and model
dataset_config = toml.load('training_scripts/config/SEED/dataset.toml')
train_data, test_data, _ = preprocess.get_dataset_SEED_split_in_train_test_validation(filename_list[0], trials_length_in_seconds,
                                                                                      dataset_config['percentage_split_train_test'], dataset_config['percentage_split_train_validation']
                                                                                      )

# Create fake labels and channels
test_labels = np.ones(len(test_data))
train_labels = np.ones(len(train_data))
ch_list = np.arange(62)

model_config = toml.load('training_scripts/config/SEED/model.toml')
model_config['encoder_config']['C'] = train_data.shape[2]
model_config['encoder_config']['T'] = train_data.shape[3] 
check_config. check_model_config_hvEEGNet(model_config)
model_hv = hvEEGNet.hvEEGNet_shallow(model_config)

# Decide if use the train or the test dataset
if use_test_set : 
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
    tmp_t = np.linspace(0, 4, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    label_name = label_dict[int(label)]
    
    # Load weight and reconstruction
    path_weight = 'Saved Model/SEED/subj_{}/model_{}.pth'.format(subj_id, epoch) # TODO Eventulmente da modificare in futuro
    path_weight = weight_file_path + 'model_{}.pth'.format(epoch)
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

    if plot_config['add_title']: ax_time.set_title('S{} - Ch. {} - Trial {}'.format(subj_id, channel, n_trial))
    
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

    if plot_config['add_title']: ax_freq.set_title('S{} - Ch. {} - Trial {}'.format(subj_id, channel, n_trial))

    fig_freq.tight_layout()
    fig_freq.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if plot_config['save_fig']:
        path_save = "Saved Results/SEED/reconstruction/subj {}/".format(subj_id)

        if use_test_set: path_save += '/test/'
        else: path_save += '/train/'

        os.makedirs(path_save, exist_ok = True)
        path_save += "s{}_trial_{}_{}_epoch_{}".format(subj_id, n_trial + 1, channel, epoch)

        if use_test_set: path_save += '_test_set'
        else: path_save += '_train_set'

        for format in plot_config['format_so_save']:
            fig_time.savefig(path_save + "_time." + format, format = format)
            fig_freq.savefig(path_save + "_freq." + format, format = format)
