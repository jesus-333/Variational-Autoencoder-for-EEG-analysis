"""
File containing various support function.
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from scipy.io import loadmat, savemat
import scipy.signal
import scipy.linalg as la

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%% Function for 100Hz dataset (Dataset IV-1) and data handling
# This function are specific for the dataset IV-1

def loadDatasetD1_100Hz(path, idx, type_dataset):
    tmp = loadmat(path + idx + '.mat');
    data = tmp['cnt'].T
    
    if(type_dataset == 'train'):
        b = tmp['mrk'][0,0]
        cue_position = b[0]
        labels  = b[1]
    else:
        cue_position = labels = None
    
    other_info = tmp['nfo'][0][0]
    sample_rate = other_info[0][0,0]
    channe_name = retrieveChannelName(other_info[2][0])
    class_label = [str(other_info[1][0, 0][0]), str(other_info[1][0, 1][0])]
    n_class = len(class_label)
    n_events = len(cue_position)
    n_channels = np.size(data, 0)
    n_samples = np.size(data, 1)
    
    other_info = {}
    other_info['sample_rate'] = sample_rate
    other_info['channel_name'] = channe_name
    other_info['class_label'] = class_label
    other_info['n_class'] = n_class
    other_info['n_events'] = n_events
    other_info['n_channels'] = n_channels
    other_info['n_samples'] = n_samples

    
    return data, labels, cue_position, other_info


def retrieveChannelName(channel_array):
    channel_list = []
    
    for el in channel_array: channel_list.append(str(el[0]))
    
    return channel_list


def computeTrialD1_100Hz(data, cue_position, labels, fs, class_label = None):
    """
    Transform the 2D data matrix of dimensions channels x samples in various 3D matrix of dimensions trials x channels x samples.
    The number of 3D matrix is equal to the number of class.
    Return everything inside a dicionary with the key label/name of the classes. If no labels are passed a progressive numeration is used.
    Parameters
    ----------
    data : Numpy matrix of dimensions channels x samples.
         Obtained by the loadDataset100Hz() function.
    cue_position : Numpy vector of length 1 x samples.
         Obtained by the loadDataset100Hz() function.
    labels : Numpy vector of length 1 x trials
        Obtained by the loadDataset100Hz() function.
    fs : int/double.
        Sample frequency.
    class_label : string list, optional
        List of string with the name of the class. Each string is the name of 1 class. The default is ['1', '2'].
    Returns
    -------
    trials_dict : dictionair
        Diciotionary with jey the various label of the data.
    """
    
    trials_dict = {}
    
    windows_sample = np.linspace(int(0.5 * fs), int(2.5 * fs) - 1, int(2.5 * fs) - int(0.5 * fs)).astype(int)
    n_windows_sample = len(windows_sample)
    
    n_channels = data.shape[0]
    labels_codes = np.unique(labels)
    
    if(class_label == None): class_label = np.linspace(1, len(labels_codes), len(labels_codes))
    
    for label, label_code in zip(class_label, labels_codes):
        # print(label)
        
        # Vector with the initial samples of the various trials related to that class
        class_event_sample_position = cue_position[labels == label_code]
        
        # Create the 3D matrix to contain all the trials of that class. The structure is n_trials x channel x n_samples
        trials_dict[label] = np.zeros((len(class_event_sample_position), n_channels, n_windows_sample))
        
        for i in range(len(class_event_sample_position)):
            event_start = class_event_sample_position[i]
            trials_dict[label][i, :, :] = data[:, windows_sample + event_start]
            
    return trials_dict

#%%

def loadDatasetD2(path, idx):
    """
    Function to load the dataset 2 of the BCI competition.
    N.B. This dataset is a costum dataset crated from the original gdf file using the MATLAB script 'dataset_transform.m'
    Parameters
    ----------
    path : string
        Path to the folder.
    idx : int.
        Index of the file.
    Returns
    -------
    data : Numpy 2D matrix
        Numpy matrix with the data. Dimensions are "samples x channel".
    event_matrix : Numpy matrix
        Matrix of dimension 3 x number of event. The first row is the position of the event, the second the type of the event and the third its duration
    """
    path_data = path + '/' + str(idx) + '_data.mat' 
    path_event = path + '/' + str(idx) + '_label.mat'
    
    data = loadmat(path_data)['data']
    event_matrix = loadmat(path_event)['event_matrix']
    
    return data, event_matrix
    
    
def computeTrialD2(data, event_matrix, fs, windows_length = 4, remove_corrupt = False, start_second = 2, end_second = 6):
    """
    Convert the data matrix obtained by loadDatasetD2() into a trials 3D matrix
    Parameters
    ----------
    data : Numpy 2D matrix
        Input data obtained by loadDatasetD2().
    event_matrix : Numpy 2D matrix
        event_matrix obtained by loadDatasetD2().
    fs: int
        frequency sampling
    windows_length: double
        Length of the trials windows in seconds. Defualt is 4.
    Returns
    -------
    trials : Numpy 3D matrix
        Matrix with dimensions "n. trials x channel x n. samples per trial".
    labels : Numpy vector
        Vector with a label for each trials. For more information read http://www.bbci.de/competition/iv/desc_2a.pdf
    """
    event_position = event_matrix[:, 0]
    event_type = event_matrix[:, 1]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Remove corrupted trials
    if(remove_corrupt):
        event_corrupt_mask_1 = event_type == 1023
        event_corrupt_mask_2 = event_type == 1023
        for i in range(len(event_corrupt_mask_2)):
            if(event_corrupt_mask_2[i] == True): 
                # Start of the trial
                event_corrupt_mask_1[i - 1] = True
                # Type of the trial
                event_corrupt_mask_1[i + 1] = True
                
        
        event_position = event_position[np.logical_not(event_corrupt_mask_1)]
        event_type = event_type[np.logical_not(event_corrupt_mask_1)]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Since trials have different length I crop them all to the minimum length
    
    # Retrieve event start
    event_start = event_position[event_type == 768]
    
    # Evaluate the samples for the trial window
    windows_sample = np.linspace(int(start_second * fs), int(end_second * fs) - 1, int(end_second * fs) - int(start_second * fs)).astype(int)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the trials matrix
    trials = np.zeros((len(event_start), data.shape[1], len(windows_sample)))
    data = data.T
    
    for i in range(trials.shape[0]):
        trials[i, :, :] = data[:, event_start[i] + windows_sample]
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the label vector
    labels = event_type[event_type != 768]
    labels = labels[labels != 32766]
    labels = labels[labels != 1023]
    
    new_labels = np.zeros(labels.shape)
    labels_name = {}
    labels_name[769] = 1
    labels_name[770] = 2
    labels_name[771] = 3
    labels_name[772] = 4
    labels_name[783] = -1
    for i in range(len(labels)):
        new_labels[i] = labels_name[labels[i]]
    labels = new_labels
    
    return trials, labels


def mapLabel(label):
    new_label = np.zeros(label.shape)
    
    for i in range(len(label)):
        el = label[i]
        
        if(el == 769): new_label[i] = 1
        if(el == 770): new_label[i] = 2
        if(el == 771): new_label[i] = 3
        if(el == 772): new_label[i] = 4
        
    return new_label


def saveTrialsForSubject(path, subject_idx, trials, labels):
    # Create the folder
    path = path + str(subject_idx) + '/'
    try: os.makedirs(path)
    except: pass
    
    # Cycle through trials 
    for i in range(len(labels)):
        trial = trials[i]
        label = labels[i]
        
        tmp_dict = {'trial': trial, 'label':label}
        
        savemat(path + str(i) + '.mat', tmp_dict)
        
        
def saveTrialsAllTogether(path, trials_list, labels_list):
    try: os.makedirs(path)
    except: pass
    
    idx = 0
    
    for trials, labels in zip(trials_list, labels_list):
        for i in range(len(labels)):
            trial = trials[i]
            label = labels[i]
        
            tmp_dict = {'trial': trial, 'label':label}
            
            savemat(path + str(idx) + '.mat', tmp_dict)
            
            idx += 1

#%% Filter and resampling function

def filterSignal(data, fs, low_f, high_f, filter_order = 3):
    # Evaluate low buond and high bound in the [0, 1] range
    low_bound = low_f / (fs/2)
    high_bound = high_f / (fs/2)
    
    # Check input data
    if(low_bound < 0): low_bound = 0
    if(high_bound > 1): high_bound = 1
    if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
    if(low_bound == high_bound): low_bound, high_bound = 0, 1
    
    b, a = scipy.signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
    
    filtered_data = scipy.signal.filtfilt(b, a, data.T)
    return filtered_data.T

    # return scipy.signal.filtfilt(b, a, data)


def resampling(trial, original_fs, new_fs, axis = 2):
    downsampling_factor = original_fs / new_fs
    signal_downsampled = scipy.signal.resample(trial, int(trial.shape[axis]/downsampling_factor), axis = axis)
    
    return signal_downsampled
        
            
#%% Exponential moving average function

def evaluateNewSample(sample, mean, variance):
    return (sample - mean) / np.sqrt(variance)


def evaluateNewMean(sample, old_mean, alpha = 0.999):
    return (1 - alpha) * sample + alpha * old_mean
    

def evaluateNewVariance(sample, mean, old_variance , alpha = 0.999):
    return (1 - alpha) * np.power(sample - mean, 2)  + alpha * old_variance


def applyExponentialAverage(data, mean_0, variance_0, alpha = 0.999):
    data_averaged = np.zeros(data.shape)
    
    # Evalaute first row with the initial mean and variance
    data_averaged[0, :] = evaluateNewSample(data[0, :], mean_0, variance_0)
    
    # First update mean and variance
    mean = evaluateNewMean(data[0, :], mean_0)
    variance = evaluateNewVariance(data[0, :], mean, variance_0)
    
    for i in range(1, data.shape[0]):
        # Evaluate new samples
        actual_row = data[i, :]
        tmp_data = evaluateNewSample(actual_row, mean, variance)
        data_averaged[i, :] = tmp_data
        
        # Update mean and variance
        mean = evaluateNewMean(data[i, :], mean)
        variance = evaluateNewVariance(data[i, :], mean, variance)
        
    return data_averaged

def applyExponentialAverageV2(data, alpha = 0.999, initial_samples = 1000):
    data = data.T
    
    filtered_data = np.zeros(data.shape)
    
    # Cycle through row (electrodes)
    for i in range(data.shape[0]):
        actual_row = data[i, :]
        
        # Evaluate initial mean and variance
        mean = np.mean(actual_row[0:initial_samples], axis = 0)
        variance = np.var(actual_row[0:initial_samples], axis = 0)
        
        # Evaluate first n_samples
        tmp_sample = evaluateNewSample(actual_row[0:initial_samples], mean, variance)
        filtered_data[i, 0:initial_samples] = tmp_sample
        
        # Cycle through columns (samples)
        for j in range(initial_samples + 1, data.shape[1]):
            # Retrieved raw samples
            actual_sample = data[i, j]
            
            # Evaluate new mean and variance
            mean = evaluateNewMean(actual_sample, mean)
            variance = evaluateNewVariance(actual_sample, mean, variance)
            
            # Evaluate new samples
            tmp_sample = evaluateNewSample(actual_sample, mean, variance)
            filtered_data[i, j] = tmp_sample

    return filtered_data.T
    
#%% 

def normalize(x, b = 1, a = 0):
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * (b - a) + a
    
    return x_norm

def standardize(x, channel_wise = False):
    
    # Perform the operation channel wise
    if(channel_wise):
        # Transpose because the original data are (samples x channels). I want (channels x samples)
        x = x.T
        x_standardize = np.zeros(x.shape)
        
        # Cycle through channel
        for i in x.shape[0]:
            # Exract channel
            channel = x[i, :]
            
            # Standardization
            tmp_new_channel = (channel - np.mean(channel)) / np.std(channel)
            
            # Save new channel
            x_standardize[i, :] = tmp_new_channel
        
        # Return the data to the original shape
        x_standardize = x_standardize.T
    else:
        # Perform the operation on the all matrix
        x_standardize = (x - np.mean(x)) / np.std(x)
    
    return x_standardize

def quantizeTrials(x, step = 255):
    possible_value = np.linspace(np.min(x), np.max(x), step)
    quantized_x = np.zeros(x.shape)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                closest_value_index = np.abs(x[i, j, k] - possible_value).argmin()
                quantized_x[i, j, k] = possible_value[closest_value_index]
    
    return quantized_x

def quantizeTrialsV2(x, step = 255):
    possible_value = np.linspace(np.min(x), np.max(x), 255)
    quantized_x = np.zeros(x.shape)
    
    for value in possible_value:
        closest_value_index = x[x - value ]

#%%

def cleanWorkspaec():
    try:
        from IPython import get_ipython
        # get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass