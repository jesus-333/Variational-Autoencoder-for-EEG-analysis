# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%% Imports 

import sys
sys.path.insert(1, 'support')

from support.support_datasets import loadDatasetD2, computeTrialD2, saveTrialsForSubject, filterSignal, resampling, normalize

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#%% Setttings

folder_save = 'v2_raw_128'

subject_idx = [1,2,3,4,5,6,7,8,9]

filtering = False
normalize_minmax = False
resampling_trials = True

low_band = 0.5
high_band = 40
new_freq = 128
start_second = 2
end_second = 6

fs = 250
path_raw_train = 'Dataset/D2A/v1/Train'
path_raw_test = 'Dataset/D2A/v1/Test'
path_SAVE_train = 'Dataset/D2A/{}/Train/'.format(folder_save)
path_SAVE_test = 'Dataset/D2A/{}/Test/'.format(folder_save)

#%%

for idx in subject_idx:
    print("Subject: ", idx)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TRAIN set
    
    data, event_matrix = loadDatasetD2(path_raw_train, idx)
    
    # Filtering
    if(filtering):
        data = filterSignal(data, fs, low_band, high_band)
        
    # Normalization minmax
    if(normalize_minmax): 
        # N.B. In this way you normalize considering all trial as a single registration so a single trial can have a max lower than 1
        data = normalize(data)
        
    # Compute trials
    trials, labels = computeTrialD2(data, event_matrix, fs, start_second = start_second, end_second = end_second)
    
    # Downsampling
    if(resampling_trials): trials =  resampling(trials, fs, new_freq, axis = 2)
    
    # From v1 to v2
    saveTrialsForSubject(path_SAVE_train, idx, trials, labels)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TEST set
    
    data, event_matrix = loadDatasetD2(path_raw_test, idx)
    
    # Filtering
    if(filtering):
        data = filterSignal(data, fs, low_band, high_band)
        
    # Normalization minmax
    if(normalize_minmax): 
        # N.B. In this way you normalize considering all trial as a single registration so a single trial can have a max lower than 1
        data = normalize(data)
        
    # Compute trials
    trials, labels = computeTrialD2(data, event_matrix, fs, start_second = start_second, end_second = end_second)
    
    # Retrieve true label
    path_true_label = 'Dataset/D2A/v1/true_labels_2a/A01E.mat'
    path_true_label = path_true_label[:-6] + str(idx) + path_true_label[-5:]
    labels = np.squeeze(loadmat(path_true_label)['classlabel'])
    
    # Downsampling
    if(resampling_trials): trials =  resampling(trials, fs, new_freq, axis = 2)
    
    # From v1 to v2
    saveTrialsForSubject(path_SAVE_test, idx, trials, labels)