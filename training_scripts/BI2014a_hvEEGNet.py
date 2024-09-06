"""
Train hvEEGNet with the dataset BI2014a.

Dataset info
link moab
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162657
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml
import numpy as np

from library.dataset import dataset_time as ds_time, download, support_function as sf
from library.training import wandb_training as wt

from library import check_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj = 1

id_machine = 'WSL_Jesus_Dell'
notes = "Nothing in particular to report"
debug = True

path_dataset_config = 'training_scripts/config/BI2014a/dataset.toml'
path_model_config = 'training_scripts/config/BI2014a/model.toml'
path_traing_config = 'training_scripts/config/BI2014a/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get config dictionaries from toml files

# Dataset
dataset_config = toml.load(path_dataset_config)
dataset_config['subjects_list'] = [subj] 

# Model and check model
model_config = toml.load(path_model_config)
check_config. check_model_config_hvEEGNet(model_config)

# Training
train_config = toml.load(path_traing_config)
train_config['train_iteration_per_subject'][str(subj)] += 1
train_config['name_training_run'] = 'S{}_{}_run_train_{}'.format(subj, id_machine, train_config['train_iteration_per_subject'][str(subj)])
train_config['debug'] = debug

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Data download and dataset creation
# (N.b. the dataloader are created inside the train_wandb_V2 function)

# Download train and test data
data_train, labels_train, ch_list = download.get_BI2014a(dataset_config, 'train')
test_data, test_labels, ch_list = download.get_Zhoug2016(dataset_config, 'test')

# Add extra dimension, necessary to work with Conv2d layer
data_train = np.expand_dims(data_train, 1)
test_data = np.expand_dims(test_data, 1)

# # For some reason the total number of samples is 1251 instead of 1250 (if no resample is used)
# (The original signal is sampled at 250Hz for 5 seconds)
# In this case to have a even number of samples the last one is removed
if dataset_config['resample_data'] == False :
    data_train = data_train[:, :, :, 0:-1]
    test_data = test_data[:, :, :, 0:-1]

# # Split train data in train and validation set
if dataset_config['percentage_split_train_validation'] > 0 and dataset_config['percentage_split_train_validation'] < 1:
    idx_train, idx_validation = sf.get_idx_to_split_data(data_train.shape[0], dataset_config['percentage_split_train_validation'], dataset_config['seed_split'])
    data_validation, labels_validation = data_train[idx_validation], labels_train[idx_validation]
    data_train, labels_train = data_train[idx_train], labels_train[idx_train]
    dataset_config['idx_train'] = idx_train
    dataset_config['idx_validation'] = idx_validation

# Get number of channels and length of time samples
C = data_train.shape[2]
T = data_train.shape[3]

# # Update model config with information from the data
model_config['encoder_config']['C'] = C
model_config['encoder_config']['T'] = T
model_config['encoder_config']['c_kernel_2'] = [C, 1]

# # Create train and validation dataset
dataset_train = ds_time.EEG_Dataset(data_train, labels_train, ch_list)
dataset_validation = ds_time.EEG_Dataset(data_validation, labels_validation, ch_list)
#
# # Train the model
# model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, dataset_train, dataset_validation)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if not debug :
    with open(path_traing_config, "w") as toml_file:
        train_config['name_training_run'] = ''
        toml.dump(train_config, toml_file)
