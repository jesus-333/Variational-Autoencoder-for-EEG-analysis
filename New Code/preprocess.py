"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function related to download the data, their preprocessing and visualization
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import mne

import moabb.datasets as mb
import moabb.paradigms as mp

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

def get_D2A_data_handmade(dataset, config, type_dataset):
    """
    Download and preprocess dataset from moabb.
    The division in trials is not handle by the moabb library but by functions that I wrote 
    """
    # Get the raw dataset for the specified list of subject
    raw_dataset = dataset.get_data(subjects = config['subjects_list'])
    
    # Iterate through subject
    for subject in config['subjects_list']:
        # Extract the data for the subject
        raw_data = raw_dataset[subject]
        
        # For each subject the data are divided in train and test. Here I extract train or test data 
        if type_dataset == 'train': raw_data = raw_data['session_T']
        elif type_dataset == 'test': raw_data = raw_data['session_E']
        else: raise ValueError("type_dataset must have value train or test")

def get_trial_handmade(raw_data, config):
    # Iterate through the run of the dataset
    for run in raw_data:
        # Extract data actual run
        raw_data_actual_run = raw_data[run]

        # Extract label and events position
        # Note that the events mark the start of a trial
        raw_info = mne.find_events(raw_data_actual_run)
        events = raw_info[:, 0]
        label = raw_info[:, 2]
        
        # Get the sampling frequency
        sampling_freq = raw_data_actual_run.info['sfreq']
        config['sampling_freq'] = sampling_freq
        
        # Filter the data
        if config['filter_data']: 
            raw_data_actual_run.filter(config['fmin'], config['fmax'])

def divide_by_event(raw_run, events, config):
    run_data = raw_run.get_data()
    trial_list = []
    for i in range(len(events) - 1):
        # Extract the data between the two events (i.e. the trial plus some data during the rest between trials)
        trial = run_data[:, events[i]:events[i + 1]]
        
        # Extract the current trial
        actual_trial = trial[0:config['sampling_freq'] * config['length_trial']]
        
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Visualization

def config_plot():
    config = dict(
        # Figure config
        figsize = (15, 10),
        # Data config
        ch_to_plot = ['C3', 'Cz', 'C4'],
        label_to_plot = [0,1,2]
    )

    return config

def visualize_single_subject_average_channel(data, label_list, ch_list):
    """
    Visualize

    data = numpy array with the EEG data. The shape must be N x C x T with N = number of trials, C = number of channels, T = time samples
    ch_list = list of length ch with the name of the channels
    label_list = numpy array of length N with the list of the label for each trial
    """
    config = config_plot()

    extracted_data = extract_data(data, ch_list, label_list, config)

    fig, ax = plt.subplots(len(config['label_to_plot']), len(config['ch_to_plot']), figsize = config['figsize'])

    for i in range(len(config['label_to_plot'])):
        for j in range(len(config['ch_to_plot'])):
            ax[i, j].plot(extracted_data[i][j])

            ax[i, j].title.set_text("{} - {}".format(config['label_to_plot'][i], config['ch_to_plot'][j]))

    fig.tight_layout()
    fig.show()

def extract_data(data, ch_list, label_list, config):
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

    
