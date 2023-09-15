"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial using the contribution of the different latent space
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

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj = 9
rand_trial_sample = True
use_test_set = True

t_min = 2
t_max = 6
fs = 250

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

train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model([subj])

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
    
    # Load weight and reconstruction (First epoch selected)
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, first_epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r_first_1, x_r_first_2, x_r_first_3 = support.compute_latent_space_different_resolution(model_hv, x.unsqueeze(0))

    # Load weight and reconstruction (Second epoch selected)
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, second_epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r_second_1, x_r_second_2, x_r_second_3 = support.compute_latent_space_different_resolution(model_hv, x.unsqueeze(0))
    
    # Select channel and time samples
    x = support.crop(x, idx_ch, 2, 6, t_min, t_max)
    x_r_first_1 = support.crop(x_r_first_1, idx_ch, 2, 6, t_min, t_max)
    x_r_first_2 = support.crop(x_r_first_2, idx_ch, 2, 6, t_min, t_max)
    x_r_first_3 = support.crop(x_r_first_3, idx_ch, 2, 6, t_min, t_max)
    x_r_second_1 = support.crop(x_r_second_1, idx_ch, 2, 6, t_min, t_max)
    x_r_second_2 = support.crop(x_r_second_2, idx_ch, 2, 6, t_min, t_max)
    x_r_second_3 = support.crop(x_r_second_3, idx_ch, 2, 6, t_min, t_max)

    # Compute PSD
    x_magnitude, x_phase, f = support.compute_spectra_magnitude_and_phase(x, fs)
    x_r_first_1_magnitude, x_r_first_1_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_1, fs)
    x_r_first_2_magnitude, x_r_first_2_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_2, fs)
    x_r_first_3_magnitude, x_r_first_3_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_3, fs)
    x_r_second_1_magnitude, x_r_second_1_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_1, fs)
    x_r_second_2_magnitude, x_r_second_2_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_2, fs)
    x_r_second_3_magnitude, x_r_second_3_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_3, fs)
