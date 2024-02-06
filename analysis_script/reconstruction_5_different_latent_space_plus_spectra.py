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

from library.analysis import support
from library.config import config_dataset as cd 

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

tot_epoch_training = 80
subj = 4
rand_trial_sample = False
use_test_set = False

t_min = 2
t_max = 3
fs = 250

nperseg = 500

plot_to_create = 20

# If rand_trial_sample == True they are selected randomly below
repetition = 1
n_trial = 145
channel = 'C5'
    
first_epoch = 10
second_epoch = 25

plot_config = dict(
    figsize = (30, 14),
    fontsize = 24,
    save_fig = False,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def plot_original_and_reconstructed(ax_list, 
                                    x_original_list, x_axis_values_list, x_r_list,
                                    x_label_list, y_label_list, title_list,
                                    ):
    for i in range(len(ax_list)):
        ax_list[i].plot(x_axis_values_list[i], x_original_list[i], label = 'Original signal')
        ax_list[i].plot(x_axis_values_list[i], x_r_list[i], label = 'Reconstructed signal')
        
        ax_list[i].legend()
        ax_list[i].grid(True)
        
        ax_list[i].set_xlabel(x_label_list[i])
        ax_list[i].set_ylabel(y_label_list[i])
        ax_list[i].set_title(title_list[i])

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
    print("Plot craeted {}/{}".format(n_plot + 1, plot_to_create))

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
    x, t = support.crop_signal(x.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_first_1, t = support.crop_signal(x_r_first_1.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_first_2, t = support.crop_signal(x_r_first_2.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_first_3, t = support.crop_signal(x_r_first_3.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_second_1, t = support.crop_signal(x_r_second_1.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_second_2, t = support.crop_signal(x_r_second_2.numpy(), idx_ch, 2, 6, t_min, t_max)
    x_r_second_3, t = support.crop_signal(x_r_second_3.numpy(), idx_ch, 2, 6, t_min, t_max)

    # Compute magnitude and phase in frequency domain
    x_magnitude, x_phase, f = support.compute_spectra_magnitude_and_phase(x, fs)
    x_r_first_1_magnitude, x_r_first_1_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_1, fs)
    x_r_first_2_magnitude, x_r_first_2_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_2, fs)
    x_r_first_3_magnitude, x_r_first_3_phase, f = support.compute_spectra_magnitude_and_phase(x_r_first_3, fs)
    x_r_second_1_magnitude, x_r_second_1_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_1, fs)
    x_r_second_2_magnitude, x_r_second_2_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_2, fs)
    x_r_second_3_magnitude, x_r_second_3_phase, f = support.compute_spectra_magnitude_and_phase(x_r_second_3, fs)
    
    
    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # PLOTs OF TYPE 1: FOR THE SAME LATENTE SPACE COMPARE TIME, FREQUENCY MAGNITUDE AND FREQUENCY PHASE AT TWO DIFFERENT EPOCH
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ALL 3 LATENT SPACE
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["Time domain - Epoch {}".format(first_epoch), "Module - Epoch {}".format(first_epoch), "Phase - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_first_1, x_r_first_1_magnitude, x_r_first_1_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["Time domain - Epoch {}".format(second_epoch), "Module - Epoch {}".format(second_epoch), "Phase - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_second_1, x_r_second_1_magnitude, x_r_second_1_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("ALL LATENT SPACE - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_1/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_1_ALL_LANTENT_SPACE_rep_{}".format(repetition)
        fig.savefig(path_save)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # DEEPER AND MIDDLE LATENT SPACE
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["Time domain - Epoch {}".format(first_epoch), "Module - Epoch {}".format(first_epoch), "Phase - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_first_2, x_r_first_2_magnitude, x_r_first_2_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["Time domain - Epoch {}".format(second_epoch), "Module - Epoch {}".format(second_epoch), "Phase - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_second_2, x_r_second_2_magnitude, x_r_second_2_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("DEEP AND MIDDLE LATENT SPACE - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_1/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_1_DEEP_AND_MIDDLE_LANTENT_SPACE_rep_{}".format(repetition)
        fig.savefig(path_save)
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # ONLY DEEPER LATENT SPACE
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["Time domain - Epoch {}".format(first_epoch), "Module - Epoch {}".format(first_epoch), "Phase - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_first_3, x_r_first_3_magnitude, x_r_first_3_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["Time domain - Epoch {}".format(second_epoch), "Module - Epoch {}".format(second_epoch), "Phase - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x, x_magnitude, x_phase], [t, f, f], [x_r_second_3, x_r_second_3_magnitude, x_r_second_3_phase],
                                    ["Time [s]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("ONLY DEEP LATENT SPACE - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
        
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_1/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_1_ONLY_DEEP_LANTENT_SPACE_rep_{}".format(repetition)
        fig.savefig(path_save)
    
    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # PLOTs OF TYPE 2: FOR THE SAME VISUALIZATION (E.G. EEG IN TIME DOMAIN) COMPARE THE THE RECONSTRUCTION ACCROSS THE DIFFERENT LATENT SPACE AT TWO DIFFERENT EPOCH
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # TIME DOMAIN
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["All - Epoch {}".format(first_epoch), "Deep and middle - Epoch {}".format(first_epoch), "Only deep - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x, x, x], [t, t, t], [x_r_first_1, x_r_first_2, x_r_first_3],
                                    ["Time [s]", "Time [s]", "Time [s]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["All - Epoch {}".format(second_epoch), "Deep and middle - Epoch {}".format(second_epoch), "Only deep - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x, x, x], [t, t, t], [x_r_second_1, x_r_second_2, x_r_second_3],
                                    ["Time [s]", "Time [s]", "Time [s]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("TIME DOMAIN - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_2/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_2_TIME_DOMAIN_rep_{}".format(repetition)
        fig.savefig(path_save)
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # MAGNITUDE
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["All - Epoch {}".format(first_epoch), "Deep and middle - Epoch {}".format(first_epoch), "Only deep - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x_magnitude, x_magnitude, x_magnitude], [f, f, f], [x_r_first_1_magnitude, x_r_first_2_magnitude, x_r_first_3_magnitude],
                                    ["Frequency [Hz]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["All - Epoch {}".format(second_epoch), "Deep and middle - Epoch {}".format(second_epoch), "Only deep - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x_magnitude, x_magnitude, x_magnitude], [f, f, f], [x_r_second_1_magnitude, x_r_second_2_magnitude, x_r_second_3_magnitude],
                                    ["Frequency [Hz]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("MAGNITUDE - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_2/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_2_MAGNITUDE_rep_{}".format(repetition)
        fig.savefig(path_save)
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # PHASE
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, axs = plt.subplots(2, 3, figsize = plot_config['figsize'])
    
    title_list = ["All - Epoch {}".format(first_epoch), "Deep and middle - Epoch {}".format(first_epoch), "Only deep - Epoch {}".format(first_epoch)]
    plot_original_and_reconstructed(axs[0], 
                                    [x_phase, x_phase, x_phase], [f, f, f], [x_r_first_1_phase, x_r_first_2_phase, x_r_first_3_phase],
                                    ["Frequency [Hz]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    title_list = ["All - Epoch {}".format(second_epoch), "Deep and middle - Epoch {}".format(second_epoch), "Only deep - Epoch {}".format(second_epoch)]
    plot_original_and_reconstructed(axs[1], 
                                    [x_phase, x_phase, x_phase], [f, f, f], [x_r_second_1_phase, x_r_second_2_phase, x_r_second_3_phase],
                                    ["Frequency [Hz]", "Frequency [Hz]", "Frequency [Hz]"], ["", "", ""], title_list,
                                    )
    
    fig.suptitle("PHASE - Trial {} - Ch {} - Label {}".format(n_trial, channel, label_dict[int(label)]), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()
    
    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/subj {}/Plot/trial_{}_channel_{}/type_2/".format(tot_epoch_training, subj, n_trial, channel, )
        os.makedirs(path_save, exist_ok = True)
        path_save += "type_2_PHASE_rep_{}".format(repetition)
        fig.savefig(path_save)
    
