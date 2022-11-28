"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Support functions related to the datasets

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from scipy.io import loadmat, savemat
import scipy.signal
from scipy.signal import resample

# from braindecode.datasets.bbci import  BBCIDataset
# from braindecode.datasets.moabb import HGD
# from braindecode.datautil.windowers import create_windows_from_events, create_fixed_length_windows

import torch
from torch import nn

#%%

def downloadDataset(idx, path_save):
    
    # Download
    a = HGD(idx)
    
    # Divide and save train and test set
    for i in range(2): 
        # i = 0 ----> Train set
        # i = 1 ----> Test  set
        tmp_dict = {0:'Train', 1:'Test'}
        print("Set: ", tmp_dict[i])
        
        dataset = a.datasets[i]
        
        dataframe_version = dataset.raw.to_data_frame()
        
        events = dataset.raw.info['events']
        
        numpy_version = dataframe_version.to_numpy()
       
        tmp_dict = {'trial': numpy_version, 'events':events}
        
        if(i == 0): savemat(path_save + 'Train/' + str(idx) + '.mat', tmp_dict)
        if(i == 1): savemat(path_save + 'Test/' + str(idx) + '.mat', tmp_dict)


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


def normalize(x, b = 1, a = 0):
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * (b - a) + a
    
    return x_norm
        
#%% HGD related function

def trialLengthDensity(idx, path_dataset): 
    # Retrieve trial start vector
    tmp_mat = loadmat(path_dataset +'/' + str(idx) + '.mat')
    trials_start_vet = tmp_mat['events'][:, 0]
    
    # Free memory
    del tmp_mat
    
    # Compute trials length
    trials_length_vet = np.zeros(len(trials_start_vet) - 1) 
    for i in range(len(trials_length_vet)):
        trials_length_vet[i] = trials_start_vet[i + 1] - trials_start_vet[i]
    
    # Find unique values and how much they appear
    tmp_unique_length = np.unique(trials_length_vet)
    trials_length_mat = np.zeros((len(tmp_unique_length), 2))
    trials_length_mat[:, 0] = tmp_unique_length
    for i in range(len(tmp_unique_length)):
        trials_length_mat[i, 1] = len(trials_length_vet[trials_length_vet == tmp_unique_length[i]])
        
    # Plot histogram
    plt.plot(trials_length_mat[:, 0], trials_length_mat[:, 1], 'o')
    plt.xlabel('# of samples')
    plt.ylabel('# of trials')


def computeTrialsHGD(idx, path_dataset, resampling = False, min_length = -1):
    
    tmp_mat = loadmat(path_dataset + str(idx) + '.mat')
    
    # Recover trials and delete first row (time) and last row (empty)
    trials = tmp_mat['trial'].T
    trials = np.delete(trials, [0, -1], 0)
    
    # Retrieve events matrix
    events = tmp_mat['events']
    
    ignore_trial = np.ones(len(events))
    if(min_length == -1): min_length = 100000000
    
    # Search the shortes trial (in number of samples) and the trial to ignore
    for i in range(len(events) - 1):
        
        start_trial_1 = events[i, 0]
        start_trial_2 = events[(i + 1), 0]
        
        # Evaluata trials length
        length_trial = start_trial_2 - start_trial_1
        
        if(length_trial > 6000): 
            # Some trials have length of 7000 or more so I will ignore them
            ignore_trial[i] = 0
        else:
            # Update min trial length
            if(length_trial < min_length and min_length != -1): 
                min_length = length_trial
                print(min_length)
    
    # Counter to number the trials        
    counter = 0
    
    # Create path for the subject
    path = path_dataset + str(idx) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Cycle to create the trials:
    for i in range(len(events)):
        if(ignore_trial[i] != 0):
            # Compute trial
            if(resampling): # Resample all trials to have min length
                start_trial_1 = events[i, 0]
                
                if(i != len(events) - 1): start_trial_2 = events[(i + 1), 0]
                else: start_trial_2 = -1
                
                # Extract trial
                tmp_trial = trials[:, start_trial_1:start_trial_2]
                
                # Matrix for the resampled trial
                resample_trial = np.zeros((tmp_trial.shape[0], min_length))
                
                # Resample to min length
                for j in range(len(trials)):
                    channel = tmp_trial[j, :]
                    channel_resampled = resample(channel, min_length)
                    resample_trial[j, :] = channel_resampled
                    
                # Substitute the matrix
                tmp_trial = resample_trial
                       
            else: # Cut all trial to have the same length
                start_trial = events[i, 0]
                tmp_trial = trials[:, start_trial:(start_trial + min_length)]
                            
            # Extract label
            tmp_label = events[i, 2]  
            
            # Save the new mat
            tmp_path = path + str(counter) + '.mat'
            tmp_dict = {'trial':tmp_trial, 'label':tmp_label}
            savemat(tmp_path, tmp_dict)
            
            counter += 1
            
            
def computeEnvelope(x, downsampling = 1):
    if(downsampling != 1): tmp_envelope = np.zeros([x.shape[0], int(x.shape[1]/downsampling)])
    else: tmp_envelope = np.zeros(x.shape)
    
    # Cycle through channels
    for j in range(x.shape[0]):
        CSP_channel = x[j, :]
        
        # Envelope evaluation
        analytic_signal = scipy.signal.hilbert(CSP_channel)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Downsampling
        if(downsampling != 1):  amplitude_envelope = scipy.signal.decimate(amplitude_envelope, downsampling)
        
        # Save the envelope
        # tmp_envelope[j, :] = amplitude_envelope[0:tmp_envelope.shape[2]]
        tmp_envelope[j, :] = amplitude_envelope
        
    return tmp_envelope



#%% D2A related function

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


#%% Dataset (PyTorch)

class PytorchDatasetEEGSingleSubject(torch.utils.data.Dataset):
    """
    Extension of PyTorch Dataset class to work with EEG data of a single subject.
    """
    
    # Inizialization method
    def __init__(self, path, n_elements = -1, normalize_trials = False, binary_mode = -1, optimize_memory = True):
        tmp_list = []
        
        # Read all the file in the folder and return them as list of string
        for element in os.walk(path): tmp_list.append(element)
        
        # print(path, tmp_list)
        self.path = tmp_list[0][0]
        self.file_list = tmp_list[0][2]
        
        self.path_list = []
        
        for i in range(len(self.file_list)): 
            file = self.file_list[i]
            self.path_list.append(path + file)
            
            if(i >= (n_elements - 1) and n_elements >= 0): break
            
        self.path_list = np.asarray(self.path_list)
        
        # Retrieve dimensions
        tmp_trial = loadmat(self.path_list[0])['trial']
        self.channel = tmp_trial.shape[0]
        self.samples = tmp_trial.shape[1]
        
        # Set binary mode
        self.binary_mode = binary_mode
        
        if(normalize_trials):
            # Temporary set to false to allow to find max and min val
            self.normalize_trials = False
            self.max_val = self.maxVal()
            self.min_val = self.minVal()
            
        # Set to real value
        self.normalize_trials = normalize_trials  
        

        
        
    def __getitem__(self, idx):
        tmp_dict = loadmat(self.path_list[idx])
        
        # Retrieve and save trial 
        trial = tmp_dict['trial']
        trial = np.expand_dims(trial, axis = 0)
        if(self.normalize_trials): trial = self.normalize(trial)
        
        # Retrieve label. Since the original label are in the range 1-4 I shift them in the range 0-3 for the NLLLoss() loss function
        label = int(tmp_dict['label']) - 1
        if(self.binary_mode != -1): 
            if(label == self.binary_mode): label = 0
            else: label = 1
               
        # Convert to PyTorch tensor
        trial = torch.from_numpy(trial).float()
        label = torch.tensor(label).long()
        
        return trial, label

    
    
    def __len__(self):
        return len(self.path_list)
    
    
    def maxVal(self):
        max_ret = -sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_max = float(torch.max(el))
            if(tmp_max > max_ret): max_ret = tmp_max
            
        return max_ret
    
    
    def minVal(self):
        min_ret = sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_min = float(torch.min(el))
            if(tmp_min < min_ret): min_ret = tmp_min
        
        return min_ret
    
    
    def normalize(self, x, a = 0, b = 1):
        x_norm = (x - self.min_val) / (self.max_val - self.min_val) * (b - a) + a
        return x_norm
    
    
class PytorchDatasetEEGMergeSubject(torch.utils.data.Dataset):
    """
    Extension of PyTorch Dataset class to work with EEG data of a multiple subjects.
    It merged the data of different subjects into a single dataset.
    """
    
    # Inizialization method
    def __init__(self, path, idx_list, n_elements = -1, normalize_trials = False, optimize_memory = True, device = 'cpu'):
        
        self.path_list = []
        
        tmp_list = []
        
        # Temporary set to True to allow some operation
        self.optimize_memory = True
        
        # Read all the file in the folder and return them as list of string
        for idx in idx_list:
            # print(path + str(idx) + '/')
            for element in os.walk(path + str(idx) + '/'): tmp_list.append(element)
        
        self.tmp_list = tmp_list
        # print(tmp_list)
        
        element_inserted = 0
        for element in tmp_list:
            for file_name in element[2]: 
                if(element_inserted >= (n_elements - 1) and n_elements >= 0): break
                
                tmp_path = element[0] + file_name
                self.path_list.append(tmp_path)
                
                element_inserted += 1
                
        # Retrieve dimensions
        tmp_trial = loadmat(self.path_list[0])['trial']
        self.channel = tmp_trial.shape[0]
        self.samples = tmp_trial.shape[1]
        
        
        if(normalize_trials):
            # Temporary set to false to allow to find max and min val
            self.normalize_trials = False
            self.max_val = self.maxVal()
            self.min_val = self.minVal()
            
        # Set to real value
        self.normalize_trials = normalize_trials 
        
        # If optimize_memory is set to false load all the dataset 
        if not optimize_memory:
            self.labels = torch.zeros(len(self.path_list))
            self.trials = torch.zeros((len(self.path_list), 1, self.channel, self.samples))
            
            for i in range(len(self.path_list)):
                tmp_trial, tmp_label = self.__getitem__(i)
                
                self.labels[i] = tmp_label
                self.trials[i, 0] = tmp_trial
                
            # Move labels and trials to device (CPU/GPU)
            self.labels = self.labels.to(device)
            self.trials = self.trials.to(device)
            
        self.optimize_memory = optimize_memory
        
        self.device = device
            
        
    def __getitem__(self, idx):
        if(self.optimize_memory):
            tmp_dict = loadmat(self.path_list[idx])
            
            # Retrieve and save trial 
            trial = tmp_dict['trial']
            trial = np.expand_dims(trial, axis = 0)
            if(self.normalize_trials): trial = self.normalize(trial)
            
            # Retrieve label. Since the original label are in the range 1-4 I shift them in the range 0-3 for the NLLLoss() loss function
            label = int(tmp_dict['label']) - 1
                    
            # Convert to PyTorch tensor
            trial = torch.from_numpy(trial).float()
            label = torch.tensor(label).long()
            
            return trial, label
        else:
            return self.trials[idx].float(), self.labels[idx].long()
    
    def __len__(self):
        return len(self.path_list)
    
    
    def maxVal(self):
        max_ret = -sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_max = float(torch.max(el))
            if(tmp_max > max_ret): max_ret = tmp_max
            
        return max_ret
    
    
    def minVal(self):
        min_ret = sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_min = float(torch.min(el))
            if(tmp_min < min_ret): min_ret = tmp_min
        
        return min_ret
    
    
    def normalize(self, x, a = 0, b = 1):
        x_norm = (x - self.min_val) / (self.max_val - self.min_val) * (b - a) + a
        return x_norm
