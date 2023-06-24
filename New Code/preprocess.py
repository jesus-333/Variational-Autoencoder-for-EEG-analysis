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

"""
%load_ext autoreload
%autoreload 2

import preprocess as pp
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download the data

def get_moabb_data_automatic(dataset, paradigm, config, type_dataset):
    """
    Return the raw data from the moabb package of the specified dataset and paradigm with some basic preprocess (implemented inside the library)
    This function utilize the moabb library to automatic divide the dataset in trials and for the baseline removal
    N.b. dataset and paradigm must be object of the moabb library
    """

    if config['resample_data']: paradigm.resample = config['resample_freq']

    if config['filter_data']: 
        paradigm.fmin = config['fmin']
        paradigm.fmax = config['fmax']
    else:
        print("NON FILTRATO")
        paradigm.fmin = 0
        paradigm.fmax = 100

    # if 'baseline' in config: paradigm.baseline = config['baseline']
    
    paradigm.n_classes = config['n_classes']

    # paradigm.tmin = -1
    # paradigm.tmax = 7.5

    # Get the raw data
    raw_data, raw_labels, info = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])
    print(info)
        
    # Select train/test data
    if type_dataset == 'train':
        idx_type = info['session'].to_numpy() == 'session_T'
    elif type_dataset == 'test':
        idx_type = info['session'].to_numpy() == 'session_E'

    raw_data = raw_data[idx_type]
    raw_labels = raw_labels[idx_type]

    return raw_data, raw_labels

def get_moabb_data_handmade(dataset, config, type_dataset):
    """
    Download and preprocess dataset from moabb.
    The division in trials is not handle by the moabb library but by functions that I wrote 
    """
    # Get the raw dataset for the specified list of subject
    raw_dataset = dataset.get_data(subjects = config['subjects_list'])
    
    # Used to save the data for each subject
    trials_per_subject = []
    labels_per_subject = []
    
    # Iterate through subject
    for subject in config['subjects_list']:
        # Extract the data for the subject
        raw_data = raw_dataset[subject]
        
        # For each subject the data are divided in train and test. Here I extract train or test data 
        if type_dataset == 'train': raw_data = raw_data['session_T']
        elif type_dataset == 'test': raw_data = raw_data['session_E']
        else: raise ValueError("type_dataset must have value train or test")

        trials_matrix, labels, ch_list = get_trial_handmade(raw_data, config)

        # Select only the data channels
        if 'BNCI2014001' in str(type(dataset)): # Dataset 2a BCI Competition IV
            trials_matrix = trials_matrix[:, 0:22, :]
            ch_list = ch_list[0:22]
        
        # Save trials and labels for each subject
        trials_per_subject.append(trials_matrix)
        labels_per_subject.append(labels)
    

    # Convert list in numpy array
    trials_per_subject = np.asarray(trials_per_subject)
    labels_per_subject = np.asarray(labels_per_subject)

    return trials_per_subject, labels_per_subject, np.asarray(ch_list) 

def get_trial_handmade(raw_data, config):
    trials_matrix_list = []
    label_list = []
    n_trials = 0

    # Iterate through the run of the dataset
    for run in raw_data:
        print(run)
        # Extract data actual run
        raw_data_actual_run = raw_data[run]

        # Extract label and events position
        # Note that the events mark the start of a trial
        raw_info = mne.find_events(raw_data_actual_run)
        events = raw_info[:, 0]
        raw_labels = raw_info[:, 2]
        # print(raw_info, "\n")
        
        # Get the sampling frequency
        sampling_freq = raw_data_actual_run.info['sfreq']
        config['sampling_freq'] = sampling_freq
        
        # (OPTIONAL) Filter data
        if config['filter_data']: raw_data_actual_run = filter_RawArray(raw_data_actual_run, config)

        # Compute trials by events
        trials_matrix_actual_run = divide_by_event(raw_data_actual_run, events, config)

        # Save trials and the corresponding label
        trials_matrix_list.append(trials_matrix_actual_run)
        label_list.append(raw_labels)
        
        # Compute the total number of trials 
        n_trials += len(raw_labels)
    
    # Convert list in numpy array
    trials_matrix = np.asarray(trials_matrix_list)
    labels = np.asarray(label_list)
    
    trials_matrix.resize(n_trials, trials_matrix.shape[2], trials_matrix.shape[3])
    labels.resize(n_trials)

    return trials_matrix, labels, raw_data_actual_run.ch_names

def filter_RawArray(raw_array_mne, config):
    # Filter the data
    filter_method = config['filter_method']
    iir_params = config['iir_params']
    if config['filter_type'] == 0: # Bandpass
         raw_array_mne.filter(l_freq = config['fmin'], h_freq = config['fmax'],
                                   method = filter_method, iir_params = iir_params)
    if config['filter_type'] == 1: # Lowpass
        raw_array_mne.filter( l_freq = None, h_freq = config['fmax'], 
                                   method = filter_method, iir_params = iir_params)
    if config['filter_type'] == 2: # Highpass 
        raw_array_mne.filter( l_freq = config['fmin'], h_freq = None, 
                                   method = filter_method, iir_params = iir_params)

    return raw_array_mne

def divide_by_event(raw_run, events, config):
    """
    Divide the actual run in trials based on the indices inside the events array
    """
    run_data = raw_run.get_data()
    trials_list = []
    for i in range(len(events)):
        # Extract the data between the two events (i.e. the trial plus some data during the rest between trials)
        if i == len(events) - 1:
            trial = run_data[:, events[i]:-1]
        else:
            trial = run_data[:, events[i]:events[i + 1]]
        
        # Extract the current trial
        actual_trial = trial[:, int(config['sampling_freq'] * config['trial_start']):int(config['sampling_freq'] * config['trial_end'])]
        trials_list.append(actual_trial)

    trials_matrix = np.asarray(trials_list)

    return trials_matrix
        
def get_data_subjects(subjecs_list : list):
    """
    Download and return the data for a list of subject
    The data are return in a matrix of size N_subject x N_trial x C x T
    """
    dataset_config = cd.get_moabb_dataset_config(subjecs_list)
    dataset = mb.BNCI2014001()
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')

    return trials_per_subject, labels_per_subject, ch_list

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

    sampling_freq = config['sampling_freq']
    

    stft_trials_matrix = []

    for i in range(trials_matrix.shape[0]): # Iterate through trials
        stft_per_channels = []
        for j in range(trials_matrix.shape[1]): # Iterate through channels
            x = trials_matrix[i, j]
            # Default
            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq / 2)

            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq, noverlap = 3 * sampling_freq / 4)
            print("nperseg = ", sampling_freq)
            print(f.shape, f)
            print(t.shape, t, "\n")

            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq / 5)
            print("nperseg = ", sampling_freq/5)
            print(f.shape, f)
            print(t.shape, t, "\n")
            
            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq / 10)
            print("nperseg = ", sampling_freq/10)
            print(f.shape, f)
            print(t.shape, t, "\n")

            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq / 25)
            print("nperseg = ", sampling_freq/25)
            print(f.shape, f)
            print(t.shape, t, "\n")
            raise ValueError("adasd")

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
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')
    
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
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')
    
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
    trials_per_subject_no_filter, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    dataset_config['filter_data'] = True 
    trials_per_subject_filter, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')

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
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    for i in range(trials_per_subject.shape[0]):
        print("Subject: {}".format(subjects_list[i]))

        trials = trials_per_subject[i]
        labels = labels_per_subject[i]

        stft_trials_matrix, t, f = compute_stft(trials, dataset_config)
        stft_trials_matrix_ERS, t = compute_ERS(stft_trials_matrix, t, f)

        pp_plot.show_after_before_ERS_on_trial(stft_trials_matrix, stft_trials_matrix_ERS, ch_list, f, t, plot_config)

