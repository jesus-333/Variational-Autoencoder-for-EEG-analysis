# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:33:56 2023

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.config import config_training as ct

from library.dataset import preprocess as pp
import library.training.train_generic as train_generic

#%%

subj_list = [[1], [2], [3], [4], [5], [6], [7], [8], [9]]
# subj_list = [[2]]
epoch = 'BEST'

results_per_subj_train = []
results_per_subj_test = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for i in range(len(subj_list)):
    print(i)
    subj = subj_list[i]
    
    dataset_config = cd.get_moabb_dataset_config(subj)
    # device = 'cpu'
    
    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    type_decoder = 0
    parameters_map_type = 0
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
    
    train_config = ct.get_config_vEEGNet_training()
    train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    test_dataloader         = torch.utils.data.DataLoader(test_dataset, batch_size = train_config['batch_size'], shuffle = True)
    
    
    # Create model
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
    model.to(device)
    
    path_weight = 'Saved Model/full_eeg/{}/model_{}.pth'.format(subj[0], epoch)
    model.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))
    
    dtw_per_trial_train = []
    for batch in train_dataloader:
        dtw_distance_matrix_train_2 = model.dtw_comparison_2(batch[0], device)
        dtw_distance_matrix_train_2 /= 1000
        tmp_dtw_per_trial = dtw_distance_matrix_train_2.mean(1)
        
        dtw_per_trial_train.append(tmp_dtw_per_trial)
        torch.cuda.empty_cache()
        
    final_results_train = np.concatenate(dtw_per_trial_train)
    results_per_subj_train.append(final_results_train)
    
    
    dtw_per_trial_test = []
    for batch in test_dataloader:
        dtw_distance_matrix_train_2 = model.dtw_comparison_2(batch[0], device)
        dtw_distance_matrix_train_2 /= 1000
        tmp_dtw_per_trial = dtw_distance_matrix_train_2.mean(1)
        
        dtw_per_trial_test.append(tmp_dtw_per_trial)
        torch.cuda.empty_cache()
        
    final_results_test = np.concatenate(dtw_per_trial_test)
    results_per_subj_test.append(final_results_test)
    
results_per_subj_train = np.asarray(results_per_subj_train)
results_per_subj_test = np.asarray(results_per_subj_test)