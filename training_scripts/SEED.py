"""
Techinical info of the SEED dataset : https://bcmi.sjtu.edu.cn/~seed/seed.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import os
import numpy as np
import toml
import scipy.io as sio
import torch

from library import check_config
from library.dataset import dataset_time as ds_time, support_function as sf, preprocess
from library.training import wandb_training as wt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
id_machine = 'MacBook_Pro_Lux'
notes = "Nothing in particular to report"
debug = True

file_path = 'data/SEED/'
subj_id = 8
# Training da fare : S4, S5, S8

sampling_freq = 200          # Defined in the link with dataset description
trials_length_in_seconds = 4 # Decided by us

path_dataset_config = 'training_scripts/config/SEED/dataset.toml'
path_model_config = 'training_scripts/config/SEED/model.toml'
path_traing_config = 'training_scripts/config/SEED/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print("CUDA backend in use")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("mps backend (apple metal) in use")
else:
    device = torch.device("cpu")
    print("No backend in use. Device set to cpu")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get config dictionaries from toml files

# Dataset config
dataset_config = toml.load(path_dataset_config)

# Model and check model
model_config = toml.load(path_model_config)
check_config.check_model_config_hvEEGNet(model_config)

# Training
train_config = toml.load(path_traing_config)
train_config['device'] = device
train_config['train_iteration_per_subject'][str(subj_id)] += 1
train_config['name_training_run'] = 'S{}_{}_run_train_{}'.format(subj_id, id_machine, train_config['train_iteration_per_subject'][str(subj_id)])
train_config['debug'] = debug

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Dataset creation

# Get all the file for a single subject
filename_list = []
list_files = os.listdir(file_path)
for file in list_files :
    if '_' in file :
        file_id = int(file.split('_')[0])
        if file_id == subj_id : filename_list.append(file_path + file)
    else :
        continue

# Load data
# For now I simply load the first file (in alphabetical order)
data = preprocess.get_dataset_SEED(filename_list[0], 4)
data = np.expand_dims(data, 1)

# Crate fake labels array
labels = np.ones(len(data))

# Split train/test
if dataset_config['percentage_split_train_test'] > 0 and dataset_config['percentage_split_train_test'] < 1: 
    # Divide in train and test set
    idx_train, idx_test = sf.get_idx_to_split_data(data.shape[0], dataset_config['percentage_split_train_test'], dataset_config['seed_split'])
    data_train, labels_train = data[idx_train], labels[idx_train]
    data_test, labels_test = data[idx_test], labels[idx_test]
else :
    raise ValueError('percentage_split_train_test in dataset_config must be between 0 and 1 for SEED dataset. Current value is {}'.format(dataset_config['percentage_split_train_test']))

# Split train data in train and validation set
if dataset_config['percentage_split_train_validation'] > 0 and dataset_config['percentage_split_train_validation'] < 1:
    idx_train, idx_validation = sf.get_idx_to_split_data(data_train.shape[0], dataset_config['percentage_split_train_validation'], dataset_config['seed_split'])
    data_validation, labels_validation = data_train[idx_validation], labels_train[idx_validation]
    data_train, labels_train = data_train[idx_train], labels_train[idx_train]
    dataset_config['idx_train'] = idx_train
    dataset_config['idx_validation'] = idx_validation
else :
    raise ValueError('percentage_split_train_validation in dataset_config must be between 0 and 1 for SEED dataset. Current value is {}'.format(dataset_config['percentage_split_train_validation']))

# Get number of channels and length of time samples
C = data_train.shape[2]
T = data_train.shape[3]

# Update model config with information from the data
model_config['encoder_config']['C'] = C
model_config['encoder_config']['T'] = T
model_config['encoder_config']['c_kernel_2'] = [C, 1]

# Create train and validation dataset
dataset_train = ds_time.EEG_Dataset(data_train, labels_train, ch_list = [])
dataset_validation = ds_time.EEG_Dataset(data_validation, labels_validation, ch_list = [])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Train the model
model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, dataset_train, dataset_validation)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if not debug :
    with open(path_traing_config, "w") as toml_file:
        train_config['name_training_run'] = ''
        toml.dump(train_config, toml_file)
