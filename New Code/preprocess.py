"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function related to download the data, their preprocessing and basic visualization
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal

import moabb.datasets as mb
import moabb.paradigms as mp
import config_dataset as cd
import config_plot as cp
import preprocess_plot as pp_plot
import download

"""
%load_ext autoreload
%autoreload 2

import preprocess as pp
"""


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Preprocess

def baseline_removal(trials_matrix, config : dict):
    stft_trials_matrix, t, f = compute_stft(trials_matrix, config)

    stft_trials_matrix_ERS, t_trial = compute_ERS(stft_trials_matrix, t, f)

    return stft_trials_matrix_ERS, t_trial, f

def compute_stft(trials_matrix, config : dict):
    """
    Compute the stft of the trials matrix channel by channel
    """

    stft_trials_matrix = []

    for i in range(trials_matrix.shape[0]): # Iterate through trials
        stft_per_channels = []
        for j in range(trials_matrix.shape[1]): # Iterate through channels
            x = trials_matrix[i, j]
            # Default
            f, t, tmp_stft = signal.stft(x, fs = config['stft_parameters']['sampling_freq'], nperseg = config['stft_parameters']['nperseg'], 
                                         window = config['stft_parameters']['window'], noverlap = config['stft_parameters']['noverlap'])
            
            stft_per_channels.append(np.power(np.abs(tmp_stft), 2))

        stft_trials_matrix.append(stft_per_channels)
    
    # Convert the list in a matrix
    stft_trials_matrix = np.asarray(stft_trials_matrix)

    # Remove the filtered frequencies
    if config['filter_data']: 
        if config['filter_type'] == 0: # Bandpass
            idx_freq = np.logical_and(f >= config['fmin'], f <= config['fmax'])
        if config['filter_type'] == 1: # Lowpass
            idx_freq = f <= config['fmax']
        if config['filter_type'] == 2: # Highpass 
            idx_freq = f >= config['fmin']
    else:
        idx_freq = np.ones(len(f)) == 1
    idx_freq = np.ones(len(f)) == 1

    return stft_trials_matrix[:, :, idx_freq, :], t, f[idx_freq]

def compute_ERS(stft_trials_matrix, t, f):
    # Indices for rest
    idx_rest = np.logical_and(t >= 0.5, t < 1.75)

    # Matrix to save the ERS
    stft_trials_matrix_ERS = np.zeros(stft_trials_matrix.shape, dtype = stft_trials_matrix.dtype)  
    
    # TODO lista dei segmenti da evitare (temporaneo)
    if stft_trials_matrix.shape[-1] == 25: k_to_avoid = [0, 23, 24]
    else: k_to_avoid = []
    k_to_avoid = []

    for i in range(stft_trials_matrix.shape[0]): # Iterate through trials
        stft_trial = stft_trials_matrix[i]
        for j in range(stft_trial.shape[0]): # Iterate through channels
            stft_channel = stft_trial[j]
            
            # Compute ERS for specific channel
            stft_trials_matrix_ERS[i, j, :] = compute_ERS_single_channels(stft_channel, idx_rest)

    return stft_trials_matrix_ERS, t

def compute_ERS_single_channels(stft_channel, idx_rest):
    stft_channel_ERS = np.zeros(stft_channel.shape)

    # Get the rest periodo and compute the average
    rest_period = stft_channel[:, idx_rest]
    average_rest_power = np.mean(rest_period, 1)

    # Compute ERS
    for k in range(stft_channel.shape[-1]):
        # stft_channel_ERS[:, k] = ( ( average_rest_power -  stft_channel[:, k]) / average_rest_power ) * 100
        stft_channel_ERS[:, k] = ( ( stft_channel[:, k] - average_rest_power) / average_rest_power ) * 100

    return stft_channel_ERS

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def download_preprocess_and_visualize():
    subjects_list = [1,2,3,4,5,6,7,8,9]
    subjects_list = [3]
    show_fig = True 

    # Download the dataset and divide it in trials
    dataset_config = cd.get_moabb_dataset_config(subjects_list)
    dataset = mb.BNCI2014001()
    trials_per_subject, labels_per_subject, ch_list = download.get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    # Config for the plots
    plot_config_random_trial = cp.get_config_plot_preprocess_random_trial() 
    plot_config_average_band = cp.get_config_plot_preprocess_average_stft() 
    plot_config_ERS = cp.get_config_plot_preprocess_ERS() 
    plot_config_random_trial['show_fig'] = plot_config_average_band['show_fig'] = plot_config_ERS['show_fig'] = show_fig
    plot_config_random_trial['t_end'] = plot_config_average_band['t_end'] = dataset_config['length_trial']
    
    # Iterate through subject and show the figures
    for i in range(trials_per_subject.shape[0]):
        print("Subject: {}".format(subjects_list[i]))
        plot_config_random_trial['subject'] = plot_config_average_band['subject'] = plot_config_ERS['subject'] = subjects_list[i]

        trials = trials_per_subject[i]
        labels = labels_per_subject[i]

        # Visualize random trial in time domain
        # plot_random_trial_time_domain(trials, ch_list, plot_config_random_trial)
        
        # Compute the ERS 
        stft_trials_matrix_ERS, t, f = baseline_removal(trials, dataset_config)

        # Correct the y limit for stft visualization 
        if dataset_config['filter_data']: 
            if dataset_config['filter_type'] == 0: # Bandpass
                plot_config_ERS['y_limit'] = [max(f[0], dataset_config['fmin']), min(f[-1], dataset_config['fmax'])]
            if dataset_config['filter_type'] == 1: # Lowpass
                plot_config_ERS['y_limit'] = [f[0], min(f[-1], dataset_config['fmax'])]
            if dataset_config['filter_type'] == 2: # Highpass 
                plot_config_ERS['y_limit'] = [max(f[0], dataset_config['fmin']), f[-1]]

        plot_config_ERS['y_limit'] = [4, 100]

        # Visualize the average band
        pp_plot.plot_average_band_stft(stft_trials_matrix_ERS, ch_list, f, plot_config_average_band)

        # Visualize the ERS 
        plot_config_ERS['t'] = t
        plot_config_ERS['f'] = f
        pp_plot.visualize_single_subject_average_channel_ERS(stft_trials_matrix_ERS, labels, ch_list, plot_config_ERS)
        
        # Close the figures
        if len(subjects_list) >= 2: plt.close('all')

def show_filter_effect():
    subjects_list = [1,2,3,4,5,6,7,8,9]
    subjects_list = [2]
    show_fig = True 

    plot_config_show_effect_filter = cp.get_config_plot_preprocess_random_trial() # The config for the plot are the same

    # Download the dataset and divide it in trials
    dataset_config = cd.get_moabb_dataset_config(subjects_list)
    dataset = mb.BNCI2014001()
    trials_per_subject, labels_per_subject, ch_list = download.get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    for i in range(trials_per_subject.shape[0]):
        print("Subject: {}".format(subjects_list[i]))

        trials = trials_per_subject[i]
        labels = labels_per_subject[i]

        plot_config_show_effect_filter['subject'] = subjects_list[i]
        dataset_config['filter_data'] = True
        pp_plot.show_filter_effect_on_trial(trials, ch_list, dataset_config, plot_config_show_effect_filter)

def show_filter_effect_on_ERS():
    subjects_list = [1,2,3,4,5,6,7,8,9]
    subjects_list = [5]
    show_fig = True 

    # Download the dataset 
    dataset_config = cd.get_moabb_dataset_config(subjects_list)
    dataset = mb.BNCI2014001()
    
    dataset_config['filter_data'] = False
    trials_per_subject_no_filter, labels_per_subject, ch_list = download.get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    dataset_config['filter_data'] = True 
    trials_per_subject_filter, labels_per_subject, ch_list = download.get_moabb_data_handmade(dataset, dataset_config, 'train')

    # Config for the plots
    plot_config_ERS = cp.get_config_plot_preprocess_ERS() 
    plot_config_ERS['show_fig'] = show_fig
    plot_config_ERS['cmap'] = 'seismic'
    
    # Iterate through subject and show the figures
    for i in range(trials_per_subject_no_filter.shape[0]):
        print("Subject: {}".format(subjects_list[i]))
        plot_config_ERS['subject'] = subjects_list[i]

        trials_no_filter = trials_per_subject_no_filter[i]
        trials_filter = trials_per_subject_filter[i]
        labels = labels_per_subject[i]
        
        # Compute the ERS 
        stft_trials_matrix_ERS_no_filter, t, f = baseline_removal(trials_no_filter, dataset_config)
        stft_trials_matrix_ERS_filter, t, f = baseline_removal(trials_filter, dataset_config)

        # Correct the y limit for stft visualization 
        if dataset_config['filter_data']: 
            if dataset_config['filter_type'] == 0: # Bandpass
                plot_config_ERS['y_limit'] = [max(f[0], dataset_config['fmin']), min(f[-1], dataset_config['fmax'])]
            if dataset_config['filter_type'] == 1: # Lowpass
                plot_config_ERS['y_limit'] = [f[0], min(f[-1], dataset_config['fmax'])]
            if dataset_config['filter_type'] == 2: # Highpass 
                plot_config_ERS['y_limit'] = [max(f[0], dataset_config['fmin']), f[-1]]

        plot_config_ERS['y_limit'] = [4, 100]
        
        vmax = 50
        vmin = -50

        plot_config_ERS['vmin'] = vmin
        plot_config_ERS['vmax'] = vmax

        stft_trials_matrix_ERS_filter[stft_trials_matrix_ERS_filter > vmax] = vmax 
        stft_trials_matrix_ERS_no_filter[stft_trials_matrix_ERS_no_filter > vmax] = vmax 

        stft_trials_matrix_ERS_filter[stft_trials_matrix_ERS_filter < vmin] = vmin
        stft_trials_matrix_ERS_no_filter[stft_trials_matrix_ERS_no_filter < vmin] = vmin 

        # Visualize the ERS 
        plot_config_ERS['t'] = t
        plot_config_ERS['f'] = f

        plot_config_ERS['filter_data'] = False
        pp_plot.visualize_single_subject_average_channel_ERS(stft_trials_matrix_ERS_no_filter, labels, ch_list, plot_config_ERS)

        plot_config_ERS['filter_data'] = True 
        pp_plot.visualize_single_subject_average_channel_ERS(stft_trials_matrix_ERS_filter, labels, ch_list, plot_config_ERS)

def show_after_before_ERS():
    subjects_list = [1,2,3,4,5,6,7,8,9]
    subjects_list = [2]
    show_fig = True 

    plot_config = cp.get_config_plot_preprocess_random_trial() # The config for the plot are the same
    plot_config['cmap'] = 'Blues_r'

    # Download the dataset and divide it in trials
    dataset_config = cd.get_moabb_dataset_config(subjects_list)
    dataset = mb.BNCI2014001()
    trials_per_subject, labels_per_subject, ch_list = download.get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    for i in range(trials_per_subject.shape[0]):
        print("Subject: {}".format(subjects_list[i]))

        trials = trials_per_subject[i]
        labels = labels_per_subject[i]

        stft_trials_matrix, t, f = compute_stft(trials, dataset_config)
        stft_trials_matrix_ERS, t = compute_ERS(stft_trials_matrix, t, f)

        pp_plot.show_after_before_ERS_on_trial(stft_trials_matrix, stft_trials_matrix_ERS, ch_list, f, t, plot_config)

