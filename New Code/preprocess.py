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

"""
%load_ext autoreload
%autoreload 2
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

    if 'baseline' in config: paradigm.baseline = config['baseline']
    
    paradigm.n_classes = config['n_classes']

    # Get the raw data
    raw_data, raw_labels, info = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])
        
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
        # Extract data actual run
        raw_data_actual_run = raw_data[run]

        # Extract label and events position
        # Note that the events mark the start of a trial
        raw_info = mne.find_events(raw_data_actual_run)
        events = raw_info[:, 0]
        raw_labels = raw_info[:, 2]
        
        # Get the sampling frequency
        sampling_freq = raw_data_actual_run.info['sfreq']
        config['sampling_freq'] = sampling_freq
        
        # Filter the data
        if config['filter_data']: 
            raw_data_actual_run.filter(config['fmin'], config['fmax'])
        
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
        actual_trial = trial[:, 0:int(config['sampling_freq'] * config['length_trial'])]
        trials_list.append(actual_trial)

    trials_matrix = np.asarray(trials_list)

    return trials_matrix
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Preprocess

def baseline_removal(trials_matrix, sampling_freq):
    stft_trials_matrix, t, f = compute_stft(trials_matrix, sampling_freq)
    stft_trials_matrix_ERS, t_trial = compute_ERS(stft_trials_matrix, t, f)

    return stft_trials_matrix_ERS, t_trial, f


def compute_stft(trials_matrix, sampling_freq):
    """
    Compute the stft of the trials matrix channel by channel
    """
    stft_trials_matrix = []
    for i in range(trials_matrix.shape[0]): # Iterate through trials
        stft_per_channels = []
        for j in range(trials_matrix.shape[1]): # Iterate through channels
            x = trials_matrix[i, j]
            f, t, tmp_stft = signal.stft(x, fs = sampling_freq, nperseg = sampling_freq / 2)

            stft_per_channels.append(np.power(np.abs(tmp_stft), 2))

        stft_trials_matrix.append(stft_per_channels)
    
    # Convert the list in a matrix
    stft_trials_matrix = np.asarray(stft_trials_matrix)

    return stft_trials_matrix, t, f

def compute_ERS(stft_trials_matrix, t, f):
    # Indices for rest
    idx_rest = np.logical_and(t >= 0.5, t < 2)

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
            
            # Get the rest periodo and compute the average
            rest_period = stft_channel[:, idx_rest]
            average_rest_power = np.mean(rest_period, 1)

            # Compute ERS
            # average_rest_power = np.tile(average_rest_power, (task_period.shape[1] , 1)).T
            # stft_trials_matrix_ERS[i, j] = ((average_rest_power - task_period) / average_rest_power) * 100
            for k in range(stft_trials_matrix_ERS.shape[-1]):
                if k not in k_to_avoid:
                    stft_trials_matrix_ERS[i, j, :, k] = ( ( average_rest_power - stft_trials_matrix[i, j, :, k] ) / average_rest_power ) * 100
                    # print(k, stft_trials_matrix_ERS[i, j, :, k].mean())

    return stft_trials_matrix_ERS, t
            
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Visualization

def plot_random_trial_time_domain(trials_matrix, ch_list, config : dict):
    """
    Select n_trials randomly for a specific channel and plot them (in time domain)

    trials_matrix = numpy array with the EEG data. The shape must be N x C x T with N = number of trials, C = number of channels, T = time samples 
    ch_list = numpy array with the name of all the channels
    """
    
    # Get the index of the channel
    idx_ch_to_plot = ch_list == config['ch_to_plot']
    
    # Numpy array with the indices of all trials
    idx_all_trials = np.arange(trials_matrix.shape[0])

    # Sample random trials to plot
    idx_trial_to_plot = np.int32(np.random.choice(idx_all_trials, config['n_trials_to_plot'], replace = False))
    trials_to_plot = trials_matrix[idx_trial_to_plot, idx_ch_to_plot]

    # Create time vector
    t = np.linspace(config['t_start'], config['t_end'], trials_to_plot.shape[-1])
    
    # Create figure
    fig, ax = plt.subplots(1,1, figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})

    # Plot the signal
    for i in range(config['n_trials_to_plot']):
        ax.plot(t, trials_to_plot[i], label = "Trial n.{}".format(idx_trial_to_plot[i]))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EEG Signal")
    ax.set_xlim([t[0], t[-1]])
    
    # Remove empty space and add legend
    fig.legend()
    fig.tight_layout()
    
    if config['save_plot']: 
        fig.savefig("Plot/eeg_time_domain_subject_{}_{}.png".format(config['subject'], config['ch_to_plot']))
    
    if config['show_fig']: fig.show()

def visualize_single_subject_average_channel_ERS(stft_data, label_list, ch_list, config):
    """
    Visualize

    stft_data =  Numpy array with the ERS of EEG data. 
            The shape must be N x C x F x T with N = number of trials, C = number of channels, F = frequencies bins of stft, T = time samples of stft
    ch_list = list of length ch with the name of the channels
    label_list = numpy array of length N with the list of the label for each trial
    """
    
    # Compute the average data for the class-channels that I want to plot
    extracted_data = extract_data_to_plot(stft_data, ch_list, label_list, config)
    
    # Get the array for x-axis and y-axis
    t = config['t']
    f = config['f']
    
    # Create figure
    fig, ax = plt.subplots(len(config['label_to_plot']), len(config['ch_to_plot']), figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})
    
    # Name of the class labels
    label_legend = {1 : 'Left', 2 : 'Right', 3 : 'Foot'}

    for i in range(len(config['label_to_plot'])):
        for j in range(len(config['ch_to_plot'])):
            # ax[i, j].plot(extracted_data[i][j])
            
            # Plot the stft
            im = ax[i, j].pcolormesh(t, f, extracted_data[i][j], shading='gouraud')

            # Label, title, limit  for each subplot
            ax[i, j].set_ylabel('Frequency [Hz]')
            ax[i, j].set_xlabel('Time [sec]')
            ax[i, j].title.set_text("{} - {}".format(label_legend[config['label_to_plot'][i]], config['ch_to_plot'][j]))
            if 'y_limit' in config: ax[i, j].set_ylim(config['y_limit'])
    
            fig.colorbar(im)

    # Title for the figure and remove empty space
    fig.suptitle('Subject {}'.format(config['subject']))
    fig.tight_layout()

    if config['save_plot']: 
        fig.savefig("Plot/ERS_Subject_{}.png".format(config['subject']))

    # Visualize figure
    if config['show_fig']: fig.show()

def extract_data_to_plot(data, ch_list, label_list, config):
    """
    Function that for each class and each channel compute the average trial (e.g. the average trial of class 'foot' for channel C3)
    """
    # List of eeg data with specific channels and average along the class
    extracted_data = []

    for label in config['label_to_plot']:
        # Get the indices of all the trial for a specific label
        idx_label = label_list == label

        # Get all the trial of the specific label and do the mean across the trial
        # The results is a matrix of shape C x T that contain the average accross the trial for a specific class
        average_per_label = data[idx_label].mean(0)
        
        # Get the channels that I want to plot
        tmp_list = []
        for ch in config['ch_to_plot']:
            idx_ch = ch_list == ch
            tmp_list.append(average_per_label[idx_ch].squeeze())
        
        # Save the list of data for the channels that I want to plot for this specific label
        extracted_data.append(tmp_list)

    return extracted_data 

def plot_average_band_stft(stft_data, ch_list, freq_array, config):
    # Get the index of the channel
    idx_ch_to_plot = ch_list == config['ch_to_plot']
    
    # Sample random trials to plot
    idx_all_trials = np.arange(stft_data.shape[0])
    idx_trial_to_plot = np.int32(np.random.choice(idx_all_trials, config['n_trials_to_average'], replace = False))
    sampled_stft = stft_data[idx_trial_to_plot, idx_ch_to_plot]
    
    # Compute the average for the specified band
    average_band_stft = compute_average_band_stft(sampled_stft, freq_array, config)

    # Create time vector
    t = np.linspace(config['t_start'], config['t_end'], len(average_band_stft))

    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    
    ax.plot(t, average_band_stft)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("STFT Average {}-{} Hz".format(config['band_start'], config['band_end']))
    ax.set_xlim([t[0], t[-1]])

    ax.set_title("Subject {} - Channel {} - Average {} trials".format(config['subject'], config['ch_to_plot'], config['n_trials_to_average']))
    
    fig.tight_layout()
    if config['save_plot']: 
        fig.savefig("Plot/Average_stft_Subject_{}_{}.png".format(config['subject'], config['ch_to_plot']))
    
    # Show figure
    if config['show_fig']: fig.show()

def compute_average_band_stft(stft_data, freq_array, config):
    # Average accross the trials
    average_stft_data = stft_data.mean(0)
    print(average_stft_data.shape)

    # Select the indices of the frequency band I'm interested
    idx_band = np.logical_and(freq_array >= config['band_start'], freq_array <= config['band_end'])
    average_band_stft = average_stft_data[idx_band, :].mean(0)
    
    return average_band_stft 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def download_preprocess_and_visualize():
    subjects_list = [1,2,3,4,5,6,7,8,9]
    subjects_list = [7]
    show_fig = True

    # Download the dataset and divide it in trials
    dataset_config = cd.get_moabb_dataset_config(subjects_list)
    dataset = mb.BNCI2014001()
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')
    
    plot_config_random_trial = cp.get_config_plot_preprocess_random_trial() 
    
    plot_config_average_band = cp.get_config_plot_preprocess_average_stft() 

    plot_config_ERS = cp.get_config_plot_preprocess_ERS() 
    plot_config_ERS['y_limit'] = [dataset_config['fmin'], dataset_config['fmax']]
    
    plot_config_random_trial['show_fig'] = plot_config_average_band['show_fig'] = plot_config_ERS['show_fig'] = show_fig
 
    for i in range(trials_per_subject.shape[0]):
        trials = trials_per_subject[i]
        labels = labels_per_subject[i]

        # Visualize random trial in time domain
        plot_config_random_trial['subject'] = subjects_list[i]
        plot_random_trial_time_domain(trials, ch_list, plot_config_random_trial)
        
        # Compute the ERS 
        stft_trials_matrix_ERS, t, f = baseline_removal(trials, dataset_config['sampling_freq'])
        
        # Visualize the average band
        plot_config_average_band['subject'] = subjects_list[i]
        plot_average_band_stft(stft_trials_matrix_ERS, ch_list, f, plot_config_average_band)

        # Visualize the ERS 
        plot_config_ERS['t'] = t
        plot_config_ERS['f'] = f
        plot_config_ERS['subject'] = subjects_list[i]
        visualize_single_subject_average_channel_ERS(stft_trials_matrix_ERS, labels, ch_list, plot_config_ERS)
        
        # Close the figures
        if len(subjects_list) >= 2: plt.close()
