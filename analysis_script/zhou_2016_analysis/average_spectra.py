#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import os

import toml
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.analysis import support

from library import check_config

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# N.b. Per ora il percorso dei pesi è hardcoded
tot_epoch_training = 80
epoch = 10
subj = 1
use_test_set = True

t_min = 1
t_max = 5

compute_spectra_with_entire_signal = True
nperseg = 500

# If rand_trial_sample == True the trial to plot are selected randomly below
rand_trial_sample = True
plot_to_create = 10

repetition = 5
n_trial = 252
channel = 'Cz'
# channel = np.random(['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz','CP4', 'O1', 'Oz', 'O2'])

plot_config = dict(
    figsize = (12, 8),
    fontsize = 18,
    save_fig = True,
    # format_so_save = ['png', 'pdf', 'eps']
    format_so_save = ['png']
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot'} # TODO CHECK

# Get subject data and model
dataset_config = toml.load('training_scripts/config/zhou2016/dataset.toml')
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
dataset_config['subjects_list'] = [subj] 
model_config = toml.load('training_scripts/config/zhou2016/model.toml')
check_config. check_model_config_hvEEGNet(model_config)
train_dataset, _, test_dataset, model_hv = support.get_dataset_zhou2016_and_model(dataset_config, model_config, model_name = 'hvEEGNet_shallow')

train_data = train_dataset.data
test_data = test_dataset.data
idx_ch = train_dataset.ch_list == channel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Train data

# Comput average spectra
average_spectra, std_spectra, f = support.compute_average_spectra(train_data, nperseg, 250, idx_ch)

# Plot
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
ax.plot(f, average_spectra, label = 'Train', color = 'black')
ax.fill_between(f, average_spectra - std_spectra, average_spectra + std_spectra, color = 'gray', alpha = 0.4)

ax.set_xlabel('Frequency [Hz]', fontsize = plot_config['fontsize'])
ax.set_ylabel(r"PSD [$\mu V^2/Hz$] (S{})".format(subj), fontsize = plot_config['fontsize'])

# Set ticks size
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])

ax.set_xlim([0, 70])
ax.set_title("S{} - Average spectra TRAIN".format(subj), fontsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/zhou2016/average_spectra/"

    os.makedirs(path_save, exist_ok = True)
    path_save += "average_spectra_s{}_train.".format(subj)

    for format in plot_config['format_so_save']:
        fig.savefig(path_save + format, format = format)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Test data

# Comput average spectra
average_spectra, std_spectra, f = support.compute_average_spectra(test_data, nperseg, 250, idx_ch)

# Plot
fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
ax.plot(f, average_spectra, label = 'Test', color = 'black')
ax.fill_between(f, average_spectra - std_spectra, average_spectra + std_spectra, color = 'gray', alpha = 0.4)

ax.set_xlabel('Frequency [Hz]', fontsize = plot_config['fontsize'])
ax.set_ylabel(r"PSD [$\mu V^2/Hz$] (S{})".format(subj), fontsize = plot_config['fontsize'])

# Set ticks size
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])

ax.set_xlim([0, 70])
ax.set_title("S{} - Average spectra TEST".format(subj), fontsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/zhou2016/average_spectra/"

    os.makedirs(path_save, exist_ok = True)
    path_save += "average_spectra_s{}_test.".format(subj)

    for format in plot_config['format_so_save']:
        fig.savefig(path_save + format, format = format)
