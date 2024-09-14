"""
Scale and shift the signal of n std, with the std compute from the train set and evaluate the reconstruction error.
Since the shift and scale change the amplitude of signal, before the error computation both the original and the reconstructed signal are scaled between 0 and 1

The formula of the trasformatino is the following x = (x * n_1 * std_train) + n_2 * std_train

Note after running the code
The formula can be rewritten as x = std_train * (n_1 * x + n_2)
This means that the effective change are proportional only to n_1 and n_2 and the moltiplication by std_train should not have great effect
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

subj = 3

epoch = 80
use_test_set = False
repetition = 5

# How many the time use the std to scale/shift the signal
n_change_step = 1
n_change_list = (np.arange(11)) * n_change_step

use_dtw_divergence = True

batch_size = 72
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

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
avg_recon_error_matrix = np.zeros((len(n_change_list), len(n_change_list)))
full_recon_error_matrix = np.zeros((len(n_change_list), len(n_change_list), raw_data.shape[0], len(train_dataset.ch_list)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute recon error for different shif/scale

for i in range(len(n_change_list)) :
    # How many the time use the std to shift the signal
    n_change_shift = n_change_list[i] 

    for j in range(len(n_change_list)) :
        # How many the time use the std to shift the signal
        n_change_scale = n_change_list[j]

        # Compute scale and shift 
        shift = float(std_train * n_change_shift)
        scale = float(std_train * n_change_scale) if n_change_scale != 0 else 1
        print("i = {}, shift = {}".format(i, shift))
        print("j = {}, scale = {}\n".format(j, scale))

        # Modify raw data and create dataset
        dataset = ds_time.EEG_Dataset((raw_data * scale) + shift, labels, train_dataset.ch_list)
        # print(raw_data.std())
        # print(dataset.data.std(), "\n")
        
        # Load model weights
        try :
            path_weight = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj, repetition, epoch)
            model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
        except :
            print("Fail to load weight subj {} epoch {} rep {}".format(subj, epoch, repetition))
        
        # Compute reconstruction error
        tmp_recon_loss = support.compute_loss_dataset(dataset, model_hv, device, batch_size, use_dtw_divergence = use_dtw_divergence, scale_before_computation = True) / 1000

        if np.sum(np.isnan(tmp_recon_loss)) > 0:
            raise ValueError("Trovato il nan per subj {} epoch {} rep {}".format(subj, epoch, repetition))
        
        # Save results
        full_recon_error_matrix[i, j] = tmp_recon_loss
        avg_recon_error_matrix[i, j] = tmp_recon_loss.mean()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Save results
path_save = 'Saved Results/d2a_analysis/shift_and_scale_error/S{}/{}/'.format(subj, string_dataset)
os.makedirs(path_save, exist_ok = True)

path_save_full = path_save + 'full_matrix_epoch_{}_rep_{}_std_{}_{}_{}.npy'.format(epoch, repetition, n_change_list[0], n_change_list[-1], n_change_step)
np.save(path_save_full, full_recon_error_matrix)

path_save_avg = path_save + 'avg_matrix_epoch_{}_rep_{}_std_{}_{}_{}.npy'.format(epoch, repetition, n_change_list[0], n_change_list[-1], n_change_step)
np.save(path_save_avg, avg_recon_error_matrix)
