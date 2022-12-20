"""
Script to create the version 2 (v2) and 3 (v3) of the dataset 2a of the BCI competition IV
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports section

import numpy as np
import os
from scipy.io import loadmat
import time
import matplotlib.pyplot as plt

from support_function import loadDatasetD2, computeTrialD2, saveTrialsForSubject, saveTrialsAllTogether
from support_function import filterSignal, resampling
from support_function import normalize, quantizeTrials, standardize, applyExponentialAverage, applyExponentialAverageV2

# from support_function_copia_github import loadDatasetD2, computeTrialD2, saveTrialsForSubject, saveTrialsAllTogether, applyExponentialAverage, applyExponentialAverageV2, filterSignal, resampling, normalize, quantizeTrials

#%% Settings
fs = 250
new_freq = 128
low_band = 4
high_band = 40
start_second = 2
end_second = 6
quantization_step = 255

exp_avg                   = False
filtering                 = True
normalize_minmax          = False
normalize_standardization = False
resampling_trials         = True
quantize_trials           = False

add_false_freq            = False
perturb_false_freq        = False 
average_perturb           = False
# false_freq_list = [55, 73, 91, 110]
false_freq_list           = [55, 75]
snr                       = 10 

folder_name = 'v2_test'
# folder_name = 'v2_raw_perturb_250_2class_snr10'

path_1_train = 'v1/Train/' 
path_2_train = '{}/Train/'.format(folder_name)
# path_3 = 'v3/'

path_1_test = 'v1/Test/'
path_2_test = '{}/Test/'.format(folder_name)


trials_list = []
labels_list = []

idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [2, 3]
idx_list = [5]

#%% Transformation

for idx in idx_list:
    print(idx)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Train set
<<<<<<< Updated upstream
    data, event_matrix = loadDatasetD2(path_1_train, idx)
=======
    
    # data, event_matrix = loadDatasetD2(path_1_train, idx)
>>>>>>> Stashed changes
    
    # # Filtering
    # if(filtering):
    #     data = filterSignal(data, fs, low_band, high_band)
    
    # # Applying expontential moving average
    # if(exp_avg):
    #     # mean_0 = np.mean(data[0:1000, :], axis = 0)
    #     # variance_0 = np.var(data[0:1000, :], axis = 0)
    #     data = applyExponentialAverageV2(data)
    
    # # Normalization minmax
    # if(normalize_minmax): 
    #     # N.B. In this way you normalize considering all trial as a single registration so a single trial can have a max lower than 1
    #     data = normalize(data)
    
    # # Normalization (x - E(X))/std(x)
    # if(normalize_standardization):
    #     data = standardize(data)
    
    # # Compute trials
    # trials, labels = computeTrialD2(data, event_matrix, fs, start_second = start_second, end_second = end_second)
    
<<<<<<< Updated upstream
    # Downsampling
    if(resampling_trials): trials =  resampling(trials, fs, new_freq, axis = 2)
=======

    # # Downsampling
    # if(resampling_trials): trials =  resampling(trials, fs, new_freq, axis = 2)
>>>>>>> Stashed changes
    
    # # Quantization
    # if(quantize_trials): trials = quantizeTrials(trials, step = quantization_step)
    
    # # Command fro v1 to v2
    # saveTrialsForSubject(path_2_train, idx, trials, labels)
    
    # print("Subject {} - Train set Finish".format(idx))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Test set
    
    # Load dataset
    data, event_matrix = loadDatasetD2(path_1_test, idx)
    
    # Filtering
    if(filtering):
        data = filterSignal(data, fs, low_band, high_band)
    
    # Applying expontential moving average
    if(exp_avg):
        mean_0 = np.mean(data[0:1000, :], axis = 0)
        variance_0 = np.var(data[0:1000, :], axis = 0)
        data = applyExponentialAverage(data, mean_0, variance_0)
     
    # Normalization minmax
    if(normalize_minmax): 
        data = normalize(data)
    
    # Compute trials
    trials, labels = computeTrialD2(data, event_matrix, fs, start_second = start_second, end_second = end_second)
    
    # Retrieve true label
    path_true_label = 'true_labels_2a/A01E.mat'
    path_true_label = path_true_label[:-6] + str(idx) + path_true_label[-5:]
    labels = np.squeeze(loadmat(path_true_label)['classlabel'])
       
    # Downsampling
    if(resampling_trials): trials =  resampling(trials, fs, new_freq, axis = 2)
    
    # Quantization
    if(quantize_trials): trials = quantizeTrials(trials, step = quantization_step)
    
    # Command fro v1 to v2
    saveTrialsForSubject(path_2_test, idx, trials, labels)
    
    print("Subject {} - Test set Finish\n".format(idx))
    
    # trials_list.append(trials)
    # labels_list.append(labels)