"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function related to download the data
"""
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import mne
import os

import moabb.datasets as mb
import moabb.paradigms as mp
from ..config import config_dataset as cd
from .. import check_config

"""
%load_ext autoreload
%autoreload 2

import download
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download and automatic segmentation (moabb)

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

    # Get the raw data
    raw_data, raw_labels, info = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])
    print(info)
        
    # Select train/test data
    if type_dataset == 'train':
        idx_type = info['session'].to_numpy() == '0train'
    elif type_dataset == 'test':
        idx_type = info['session'].to_numpy() == '1test'

    raw_data = raw_data[idx_type]
    raw_labels = raw_labels[idx_type]

    return raw_data, raw_labels

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download and  segmentation through mne

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
        if type_dataset == 'train': raw_data = raw_data['0train']
        elif type_dataset == 'test': raw_data = raw_data['1test']
        else: raise ValueError("type_dataset must have value train or test")

        trials_matrix, labels, ch_list = get_trial_handmade(raw_data, config)

        # Select only the data channels
        if 'BNCI2014_001' in str(type(dataset)): # Dataset 2a BCI Competition IV
            if 'channels_list' in config: idx_ch = get_idx_ch(ch_list, config)
            else: idx_ch = np.arange(22)
            
            trials_matrix = trials_matrix[:, idx_ch, :]
            ch_list = ch_list[idx_ch]
            
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

    return trials_matrix, labels, np.asarray(raw_data_actual_run.ch_names)

def filter_RawArray(raw_array_mne, config):
    # Filter the data
    filter_method = config['filter_method']
    iir_params = config['iir_params']
    if config['filter_type'] == 0: # Bandpass
        raw_array_mne.filter(l_freq = config['fmin'], h_freq = config['fmax'],
                          method = filter_method, iir_params = iir_params)
    if config['filter_type'] == 1: # Lowpass
        raw_array_mne.filter(l_freq = None, h_freq = config['fmax'],
                             method = filter_method, iir_params = iir_params)
    if config['filter_type'] == 2: # Highpass
        raw_array_mne.filter(l_freq = config['fmin'], h_freq = None,
                             method = filter_method, iir_params = iir_params)
    if config['filter_type'] == 3: # Notch Filter
        raw_array_mne.filter(freqs = config['notch_freq'],
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
    
    # The 1e6 factor is used to scale the input signal (the original signal is in microvolt)
    trials_matrix = np.asarray(trials_list) * 1e6

    return trials_matrix

def get_idx_ch(ch_list_dataset, config):
    """
    Function to create a list of indices to select only specific channels
    """
    
    idx_ch = np.zeros(len(config['channels_list']), dtype = int)
    for i in range(len(config['channels_list'])):
        ch_to_use = config['channels_list'][i]
        for j in range(len(ch_list_dataset)):
            ch_dataset = ch_list_dataset[j]
            if ch_to_use == ch_dataset: idx_ch[i] = int(j)
            
    return idx_ch
        
def get_data_subjects_train(subjecs_list : list):
    """
    Download and return the data for a list of subject
    The data are return in a numpy array of size N_subject x N_trial x C x T
    """
    dataset_config = cd.get_moabb_dataset_config(subjecs_list)
    dataset = mb.BNCI2014_001()
    trials_per_subject, labels_per_subject, ch_list = get_moabb_data_handmade(dataset, dataset_config, 'train')

    return trials_per_subject, labels_per_subject, ch_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_D2a_data(config, type_dataset):
    check_config.check_config_dataset(config)
    mne.set_log_level(False)
    
    # Select the dataset
    dataset = mb.BNCI2014_001()

    # Select the paradigm (i.e. the object to download the dataset)
    paradigm = mp.MotorImagery()

    # Get the data
    if config['use_moabb_segmentation']:
        raw_data, raw_labels = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)
        
        # Select channels and convert labels
        # N.b. since for now we work only with dataset 2a I hardcode the 22 channels selection
        data = raw_data[:, 0:22, :]
        labels = convert_label(raw_labels)

        ch_list = get_dataset_channels(dataset)[0:22]
    else:
        data, labels, ch_list = get_moabb_data_handmade(dataset, config, type_dataset)

        # With the "handmade" division the data are returned in shape: "n. subject x trials x channels x time samples"
        # This operation remove the first dimension
        data = data.reshape(-1, data.shape[2], data.shape[3])
        labels = labels.reshape(-1)
        
        # By default the labels obtained through the moabb have value between 1 and 4.
        # But Pytorch for 4 classes want values between 0 and 3
        labels -= 1

    return data, labels.squeeze(), ch_list

def get_dataset_channels(dataset):
    """
    Get the list of channels for the specific dataset
    """

    if 'BNCI2014_001' in str(type(dataset)) : # Dataset 2a BCI Competition IV
        raw_data = dataset.get_data(subjects = [1])[1]['0train']['run_0']
        ch_list = raw_data.ch_names
    elif 'RawEDF' in str(type(dataset)) :
        ch_list = dataset.ch_names
    else:
        raise ValueError("Function not implemented for this type of dataset")

    return np.asarray(ch_list)

def convert_label(raw_labels, use_BCI_D2a_label = True):
    """
    Convert the "raw" label obtained from the moabb dataset into a numerical vector where to each label is assigned a number
    use_BCI_D2a_label is a parameter that assign for the label of the Dataset 2a of BCI Competition IV specific label
    """
    
    # Create a vector of the same lenght of the previous labels vector
    new_labels = np.zeros(len(raw_labels))
    
    # Create a list with all the possible labels
    if use_BCI_D2a_label:
        labels_list = dict(
            left_hand = 1,
            right_hand = 2,
            feet = 3,
            tongue = 4,
        )
    else:
        labels_list = np.unique(raw_labels)
    
    # Iterate through the possible labels
    if use_BCI_D2a_label:
        for label in labels_list:
            print("Label {} get the value {}".format(label, labels_list[label]))
            idx_label = raw_labels == label
            new_labels[idx_label] = int(labels_list[label])
    else:
        for i in range(len(labels_list)):
            print("Label {} get the value {}".format(labels_list[i], i))

            # Get the label
            label = labels_list[i]
            idx_label = raw_labels == label
            
            # Assign numerical label
            new_labels[idx_label] = int(i)

    return new_labels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# TUAR Dataset (Note that this is not the entire dataset but a subsection hosted on google drive)

def get_TUAR_data(config : dict) :
    # Check if the data are already downloaded
    if os.path.exists('data/TUAR_dataset') :
        if 'force_download' not in config : config['force_download'] = False
        
        # If force_download is set to True download the dataset anyway
        if config['force_download'] :
            print('Download TUAR dataset')
            download_TUAR()
        else :
            print('The dataset is already downloaded.')

    data_raw = mne.io.read_raw_edf(config['path_file'])
    ch_list = get_dataset_channels(data_raw)

    return data_raw, ch_list

def download_TUAR() :
    # Link to shared file in google drive and path to save the data
    link_drive = 'https://drive.google.com/drive/folders/10JmpvbkcMVc20EJt0caAqBDxYOkZth31?usp=sharing'
    path_save = 'data/'

    download_from_google_drive(link_drive, path_save)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def download_from_google_drive(link_drive : str, path_save : str) :
    try :
        # Import the package to download stuff from gooogle drive
        # The import is here to avoid people to force install googledriver if they will not use TUAR data
        from googledriver import download_folder
        
        # Check that the folder to save the data exist, otherwise create it
        os.makedirs(path_save, exist_ok = True)

        # Download the folder with the TUAR dataset
        download_folder(link_drive, path_save)
    except :
        raise ImportError("To download the TUAR dataset you need the googledriver package. You can download it from https://pypi.org/project/googledriver/")
