"""
Shift the signal of n std, with the std compute from the train set and evaluate the reconstruction error.
The signal shift is computed as : x = x + n * std_train
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import os

from library.config import config_dataset as cd
from library.dataset import dataset_time as ds_time
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj = 6
subj_list = [7, 8, 9]

epoch = 80
use_test_set = False
repetition_list = [8, 9]

# How many the time use the std to scale/shift the signal
n_change_list = np.arange(16) 

use_dtw_divergence = True

batch_size = 72
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
for subj in subj_list :
    print("Subject : ", subj)
    # Get the datasets
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset

    # Get data
    train_dataset, _, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')
    if use_test_set :
        raw_data = test_dataset.data.numpy()
        labels = test_dataset.labels.numpy()
        string_dataset = 'test'
    else :
        raw_data = train_dataset.data.numpy()
        labels = train_dataset.labels.numpy()
        string_dataset = 'train'

    # Compute std of the training set
    std_train = train_dataset.data.std(-1).mean()

    # Variable to save results
    full_recon_error_matrix = np.zeros((len(n_change_list), raw_data.shape[0], len(train_dataset.ch_list)))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute recon error for different shif/scale

    for repetition in repetition_list :
        print("\tRepetition : ", repetition)
        for i in range(len(n_change_list)) :
            # How many the time use the std to shift the signal
            n_change_shift = n_change_list[i] 

            # Compute scale and shift 
            shift = float(std_train * n_change_shift)
            print("\t\tn = {} (i = {}), shift = {}".format(n_change_shift, i , shift))

            # Modify raw data and create dataset
            dataset = ds_time.EEG_Dataset(raw_data + shift, labels, train_dataset.ch_list)
            
            # Load model weights
            try :
                path_weight = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj, repetition, epoch)
                model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
            except :
                print("Fail to load weight subj {} epoch {} rep {}".format(subj, epoch, repetition))
            
            # Compute reconstruction error
            tmp_recon_loss = support.compute_loss_dataset(dataset, model_hv, device, batch_size, use_dtw_divergence = use_dtw_divergence, scale_before_computation = True) / 1000
            full_recon_error_matrix[i] = tmp_recon_loss

            if np.sum(np.isnan(tmp_recon_loss)) > 0:
                print("Trovato il nan per subj {} epoch {} rep {}".format(subj, epoch, repetition))
                continue
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Save results
            path_save = 'Saved Results/d2a_analysis/shift_error/S{}/{}/'.format(subj, string_dataset)
            os.makedirs(path_save, exist_ok = True)

            path_save_full = path_save + 'recon_matrix_epoch_{}_rep_{}_std_{}.npy'.format(epoch, repetition, n_change_shift)
            np.save(path_save_full, tmp_recon_loss)

