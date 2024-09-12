"""
Shift the signal of n std, with the std compute from the train set and evaluate the reconstruction error.
Since the shift change the scale of signal, before the error computation both the original and the reconstructed signal are scaled between 0 and 1
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np

from library.config import config_dataset as cd
from library.dataset import dataset_time as ds_time
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj = 3

epoch = 55
use_test_set = False
repetition = 5

# How many the time use the std to scale/shift the signal
n_change_step = 1
n_change_list = (np.arange(2) + 1) * n_change_step

batch_size = 64
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
else :
    raw_data = train_dataset.data.numpy()
    labels = train_dataset.labels.numpy()

# Compute std of the training set
std_train = train_dataset.data.std(-1).mean()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute recon error for different shif/scale

for i in range(len(n_change_list)) :
    # How many the time use the std to shift the signal
    n_change_shift = n_change_list[i] 

    for j in range(len(n_change_list)) :
        # How many the time use the std to shift the signal
        n_change_scale = n_change_list[j] - n_change_step

        # Compute scale and shift 
        shift = float(std_train * n_change_shift)
        scale = float(std_train * n_change_scale) if n_change_scale != 0 else 1

        # Modify raw data and create dataset
        dataset = ds_time.EEG_Dataset((raw_data + shift) * scale, labels, train_dataset.ch_list)

        print(shift, scale)
        print(raw_data.mean())
        print(dataset.data.mean(), "\n")
        
        # Load model weights
        try :
            path_weight = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj, repetition, epoch)
            model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
        except :
            print("Fail to load weight subj {} epoch {} rep {}".format(subj, epoch, repetition))
        
        # Compute reconstruction error
        # tmp_recon_loss = support.compute_loss_dataset(dataset, model_hv, device, batch_size) / 1000

        # if np.sum(np.isnan(tmp_recon_loss)) > 0:
        #     raise ValueError("Trovato il nan per subj {} epoch {} rep {}".format(subj, epoch, repetition))
