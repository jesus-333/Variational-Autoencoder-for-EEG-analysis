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
from . import support_function as sf
from ..config import config_dataset as cd
from .. import check_config

"""
%load_ext autoreload
%autoreload 2

import download
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download and automatic segmentation (moabb)

def get_moabb_data_automatic(dataset, paradigm, config : dict, type_dataset : str):
    """
    Return the raw data from the moabb package of the specified dataset and paradigm with some basic preprocess (implemented inside the library)
    This function utilize the moabb library to automatic divide the dataset in trials and for the baseline removal
    N.b. dataset and paradigm must be object of the moabb library
    """

    check_config.check_config_dataset(config)
    mne.set_log_level(False)

    if config['resample_data']: paradigm.resample = config['resample_freq']

    if config['filter_data']: 
        paradigm.fmin = config['fmin']
        paradigm.fmax = config['fmax']
    else:
        print("No filter applied")

    # if 'baseline' in config: paradigm.baseline = config['baseline']
    
    paradigm.n_classes = config['n_classes']
    
    # Set start and end of each trial
    if config['trial_start'] > 0 and config['trial_end'] > config['trial_start'] :
        paradigm.tmin = config['trial_start']
        paradigm.tmax = config['trial_end']
    else : 
        print("Invalid values or not specified for trial_start and trials_end in the config. Used default values provided by moabb.")

    # Get the raw data
    raw_data, raw_labels, info = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])

    # Select train/test data
    idx_type = get_idx_train_or_test(dataset, info, type_dataset)
    data = raw_data[idx_type]
    raw_labels = raw_labels[idx_type]

    # Select channels and convert labels
    labels = convert_label(raw_labels, use_BCI_D2a_label = False) 
    
    # Get channels list
    ch_list = get_dataset_channels(dataset) 

    return data, labels, ch_list

def get_idx_train_or_test(dataset, info : dict, type_dataset : str) :
    """
    Inside the MOABB every dataset can have different label to indicate which trials are train or test.
    The label are saved inside the info variable returned from paradigm.get_data()

    @param dataset : dataset object, from MOABB library
    @param info : (dict) Dictionary obtained from paradig.get_data()
    @param type_dataset : (str) String that specify which type of data obtain, i.e. train, test or full. 

    Note on type_dataset : Some dataset have not a division between train and test. For now if you specified train or test this function simply divides the dataset in two, with the first half as train data and the second as test data.
                           If you set type_dataset = full, ONLY FOR THIS DATASET, it will return the entire dataset.
    Dataset without a diviosn in train/test : Weibo2014, Cho2017, BI2014a

    @return idx_type : (numpy array) Array of boolean that specified which trails are for train or test
    """

    name_dataset = str(type(dataset))
    
    if '2014_001' in name_dataset or '2014001' in name_dataset:
        if   type_dataset == 'train': idx_type = info['session'].to_numpy() == '0train'
        elif type_dataset == 'test' : idx_type = info['session'].to_numpy() == '1test'
    elif 'Zhou2016' in name_dataset : # Zhou2016
        # This list contains the id of each run for each trial
        run_of_each_trials = info['run'].to_numpy()
        
        # Note that a certain point the id used to to distinguish between run of type 0 and 1 was changed.
        # To keep compatibility with each possible version of moabb I check which id is used
        if type_dataset == 'train':
            if '0' in run_of_each_trials : idx_type = run_of_each_trials == '0'
            elif 'run_0' in run_of_each_trials : idx_type = run_of_each_trials == 'run_0'
            else : raise ValueError("Probably there is some problem with the wandb version")
        elif type_dataset == 'test':
            if '1' in run_of_each_trials : idx_type = run_of_each_trials == '1'
            elif 'run_1' in run_of_each_trials : idx_type = run_of_each_trials == 'run_1'
            else : raise ValueError("Probably there is some problem with the wandb version")
    elif 'Lee2019_MI' in name_dataset : # Zhou2016
        # This list contains the id of each run for each trial
        session_of_each_trials = info['session'].to_numpy()
        
        # Note that a certain point the id used to to distinguish between run of type 0 and 1 was changed.
        # To keep compatibility with each possible version of moabb I check which id is used
        if type_dataset == 'train':
            if '0' in session_of_each_trials : idx_type = session_of_each_trials == '0'
            elif 'session_0' in session_of_each_trials : idx_type = session_of_each_trials == 'session_0'
            else : raise ValueError("Probably there is some problem with the wandb version")
        elif type_dataset == 'test':
            if '1' in session_of_each_trials : idx_type = session_of_each_trials == '1'
            elif 'session_1' in session_of_each_trials : idx_type = session_of_each_trials == 'session_1'
    elif 'Ofner2017' in name_dataset : # Ofner2017 
        # In the Ofner2017 dataset for each subject there are 10 sessions of recording (with idx from 0 to 9)
        # I take the first 5 sessions as training data and the other 5 as test data
        # If I set type_dataset = full it will return all the sessions
        idx_type = np.ones(len(info['run'].to_numpy())) == 0
        if type_dataset == 'train':
            for i in [0, 1, 2, 3, 4] : idx_type += info['run'].to_numpy() == str(i)
        elif type_dataset == 'test':
            for i in [5, 6, 7, 8, 9] : idx_type += info['run'].to_numpy() == str(i)
        elif type_dataset == 'full' : 
            idx_type = np.ones(len(info['run'].to_numpy())) == 1
    elif 'Schirrmeister2017' in name_dataset : # Schirrmeister2017 
        if   type_dataset == 'train': idx_type = info['run'].to_numpy() == '0train'
        elif type_dataset == 'test' : idx_type = info['run'].to_numpy() == '1test'
    elif 'BI2014a'   in name_dataset or \
         'Weibo2014' in name_dataset or \
         'Cho2017'   in name_dataset or \
         'GrosseWentrup2009' in name_dataset : 

        # In these datasets train and test are not indicated. So for now I take half of the dataset for training and the other half for test
        # If I pass the full option it return the entire dataset
        n_elements = len(info['run'].to_numpy())
        if type_dataset == 'train'  : idx_type = np.arange(0, int(n_elements / 2))
        elif type_dataset == 'test' : idx_type = np.arange(int(n_elements / 2), n_elements)
        elif type_dataset == 'full' : idx_type = np.arange(n_elements)
    else :
        raise ValueError('Dataset not supported')

    return idx_type

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
        if type_dataset == 'train' : raw_data = raw_data['0train']
        elif type_dataset == 'test' : raw_data = raw_data['1test']
        else : raise ValueError("type_dataset must have value train or test")

        trials_matrix, labels, ch_list = get_trial_handmade(raw_data, config)

        # Select only the data channels
        if 'BNCI2014_001' in str(type(dataset)): # Dataset 2a BCI Competition IV
            if 'channels_list' in config and len(config['channels_list']) > 0 : idx_ch = get_idx_ch(ch_list, config)
            else : idx_ch = np.arange(22)
            
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
        # Note that for some reasons the automatic division of the dataset by the moabb created  281 trials, insted of 288.

        raw_data, raw_labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)
        
        # Select channels and convert labels
        # Note that d2a has 25 channels but the last 3 where for electro-oculogram and thus are excluded
        data = raw_data[:, 0:22, :]
        labels = convert_label(raw_labels)
        ch_list = ch_list[0:22]
    else:
        data, labels, ch_list = get_moabb_data_handmade(dataset, config, type_dataset)

        # With the "handmade" division the data are returned in shape: "n. subject x trials x channels x time samples"
        # This operation remove the first dimension
        data = data.reshape(-1, data.shape[2], data.shape[3])
        labels = labels.reshape(-1)
        
        # By default the labels obtained through the moabb have value between 1 and 4.
        # But Pytorch for 4 classes want values between 0 and 3
        labels -= 1
    
    # (OPTIONAL) if use_fewer_trials is a positive number randomly select that number of trials
    if config['use_fewer_trials'] > 0:
        np.random.seed(config['seed_split'])
        idx = np.random.permutation(data.shape[0])
        idx = idx[0:config['use_fewer_trials']]
        data = data[idx]
        labels = labels[idx]

    return data, labels.squeeze(), ch_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Motor Imagery datasets

def get_Zhou2016(config : dict, type_dataset : str) :
    """
    Throuhg moabb, get the dataset preseneted by Zhou et al. 2016 (https://doi.org/10.1371/journal.pone.0162657)

    @param config : (dict) Dictionary with the config for the dataset
    @param type_dataset : (str) String that must have values train or test. Specify if returning the training or test data

    @return data : (numpy array) Numpy array with shape N x C x T, with N = n. of trials, C = n. of channels, T = n. of time samples.
    @retunr label : ...
    @return ch_list : (list) List with the name of the channels
    """
    check_config.check_config_dataset(config)
    mne.set_log_level(False)
    
    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Zhou2016()
    paradigm = mp.MotorImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)
    
    # Note that Zhou2016 has 14 channels but the first 2 are for electro-oculogram (VEOU, VEOUL) and thus are excluded
    # From the data this 2 channel are automatically removed by the paradigm.get_data() function
    ch_list = ch_list[2:] 

    return data, labels.squeeze(), ch_list

def get_Weibo2014(config : dict, type_dataset : str) :
    """
    Throuhg moabb, get the dataset preseneted by ....

    @param config : (dict) Dictionary with the config for the dataset
    @param type_dataset : (str) String that must have values train or test. Specify if returning the training or test data

    @return data : (numpy array) Numpy array with shape N x C x T, with N = n. of trials, C = n. of channels, T = n. of time samples.
    @retunr label : ...
    @return ch_list : (list) List with the name of the channels
    """
    
    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Weibo2014()
    paradigm = mp.MotorImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    # The last 3 channels (VEO, HEO, STIMO14) are excluded from the data by the paradigm so I remove them from the channels list
    ch_list = ch_list[0:-3] 

    return data, labels.squeeze(), ch_list

def get_Cho2017(config : dict, type_dataset : str) :
    """
    Get the dataset presented by Cho et al. 2017 (https://doi.org/10.1093/gigascience/gix034)
    """

    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Cho2017()
    paradigm = mp.LeftRightImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    # Remove the last 5 channels, automatically removed from the data by the paradigm (EMG1, EMG2, EMG3, EMG4, Stim)
    ch_list = ch_list[0:-5]

    return data, labels.squeeze(), ch_list

def get_Ofner2017(config : dict, type_dataset : str) :
    """
    Get the dataset presented by 
    """

    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Ofner2017()
    paradigm = mp.MotorImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    # Rescale data (The original data are in the range 10^6-10^7)
    data = data * (10 ** -6)

    # Remove from the channels list that the paradigm automatically removed
    ch_list = ch_list[0:-35]

    return data, labels.squeeze(), ch_list

def get_GrosseWentrup2009(config : dict, type_dataset : str) :
    """
    Get the dataset presented by 
    """

    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.GrosseWentrup2009()
    paradigm = mp.LeftRightImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    return data, labels.squeeze(), ch_list

def get_Lee2019_MI(config : dict, type_dataset : str) :
    """
    Get the dataset presented by 
    """

    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Lee2019_MI()
    paradigm = mp.LeftRightImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    # Remove the last 5 channels, automatically removed from the data by the paradigm (EMG1, EMG2, EMG3, EMG4, STI 014)
    ch_list = ch_list[0:-5]

    return data, labels.squeeze(), ch_list

def get_Schirrmeister2017(config : dict, type_dataset : str) :
    """
    Get the dataset presented by 
    """

    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.Schirrmeister2017()
    paradigm = mp.MotorImagery()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)

    ch_list = ch_list

    return data, labels.squeeze(), ch_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# P300 Datasets

def get_BI2014a(config : dict, type_dataset : str) :
    """
    Throuhg moabb, get the dataset preseneted by Huebner et al. 2018 (https://doi.org/10.1109/MCI.2018.2807039)

    @param config : (dict) Dictionary with the config for the dataset
    @param type_dataset : (str) String that must have values train or test. Specify if returning the training or test data

    @return data : (numpy array) Numpy array with shape N x C x T, with N = n. of trials, C = n. of channels, T = n. of time samples.
    @retunr label : ...
    @return ch_list : (list) List with the name of the channels
    """
    # Select the dataset and paradigm (i.e. the object to download the dataset)
    dataset = mb.BI2014a()
    paradigm = mp.P300()
    
    # Get data, labels and channels list
    data, labels, ch_list = get_moabb_data_automatic(dataset, paradigm, config, type_dataset)
    ch_list = ch_list[0:-1] 

    return data, labels.squeeze(), ch_list

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
    else :
        download_TUAR()

    # Check the config
    check_config.check_config_dataset(config)
    mne.set_log_level(False)
    
    # Read data
    data_raw = mne.io.read_raw_edf(config['path_file'])
    
    # Preprocess data
    if 'channels_to_use' in config : data_raw.pick_channels(config['channels_to_use'], ordered = True)  # Reorders the channels and drop the ones not contained in channels_to_use
    if config['resample_data'] : data_raw.resample(config['resample_freq'])                             # Resample data
    if config['filter_data']   : data_raw = filter_RawArray(data_raw, config)                           # (OPTIONAL) Filter the data
    
    # Segmentation
    data_segmented = mne.make_fixed_length_epochs(data_raw, duration = config['segment_duration'], preload = False, overlap = config['segment_overlap_duration'])
    data_segmented = data_segmented.get_data()

    # Get channels list
    ch_list = get_dataset_channels(data_raw)
    
    # TODO (?)
    if 'samples_to_use' in config : data_segmented = data_segmented[0:config['samples_to_use']]

    return data_segmented, data_raw, ch_list

def download_TUAR() :
    """
    Download some file of the TUAR dataset from google drive
    """
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other functions

def get_dataset_channels(dataset):
    """
    Get the list of channels for the specific dataset
    """

    if 'BNCI2014_001' in str(type(dataset)) : # Dataset 2a BCI Competition IV
        raw_data = dataset.get_data(subjects = [1])[1]['0train']['run_0']
        ch_list = raw_data.ch_names
    elif 'RawEDF' in str(type(dataset)) :
        ch_list = dataset.ch_names
    elif 'Zhou2016'  in str(type(dataset)) or \
         'BI2014a'   in str(type(dataset)) or \
         'Weibo2014' in str(type(dataset)) or \
         'Cho2017'   in str(type(dataset)) or \
         'GrosseWentrup2009' in str(type(dataset)) :

        raw_data = dataset.get_data(subjects = [1])[1]

        if '0' in raw_data : raw_data = raw_data['0']['0']
        elif 'session_0' in raw_data : raw_data = raw_data['session_0']['run_0']
        else : raise ValueError('Probably there is some problem with the moabb version')
       
        ch_list = raw_data.ch_names
    elif 'Ofner2017' in str(type(dataset)) :
        ch_list = dataset.get_data(subjects = [1])[1]['1imagination']['0'].ch_names
    elif 'Lee2019_MI' in str(type(dataset)) :
        ch_list = dataset.get_data(subjects = [1])[1]['0']['1train'].ch_names
    elif 'Schirrmeister2017' in str(type(dataset)) :
        ch_list = dataset.get_data(subjects = [1])[1]['0']['0train'].ch_names
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
