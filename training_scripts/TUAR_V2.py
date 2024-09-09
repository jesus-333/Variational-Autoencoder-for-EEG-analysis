"""
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import toml

from library import check_config
from library.dataset import dataset_time as ds_time, support_function as sf
from library.training import wandb_training as wt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
id_machine = 'WSL_Jesus_Dell'
notes = "Nothing in particular to report"
debug = True

filename = 'NO_NOTCH_train8'

path_dataset_config = 'training_scripts/config/TUAR/dataset.toml'
path_model_config = 'training_scripts/config/TUAR/model.toml'
path_traing_config = 'training_scripts/config/TUAR/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get config dictionaries from toml files

# Dataset config
dataset_config = toml.load(path_dataset_config)

# Model and check model
model_config = toml.load(path_model_config)
check_config.check_model_config_hvEEGNet(model_config)

# Training
train_config = toml.load(path_traing_config)
train_config['train_iteration_per_subject'][str(filename)] += 1
train_config['name_training_run'] = '{}_{}_run_train_{}'.format(filename, id_machine, train_config['train_iteration_per_subject'][str(filename)])
train_config['debug'] = debug

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Dataset creation

# Load data
data_train = np.load('data/TUAR/NO_NOTCH_train8.npz')['train_data']
data_test = np.load('data/TUAR/NO_NOTCH_train8.npz')['test_data']

# Crate fake labels array
labels_train = np.ones(len(data_train))
labels_test = np.ones(len(data_test))

# Split train data in train and validation set
if dataset_config['percentage_split_train_validation'] > 0 and dataset_config['percentage_split_train_validation'] < 1:
    idx_train, idx_validation = sf.get_idx_to_split_data(data_train.shape[0], dataset_config['percentage_split_train_validation'], dataset_config['seed_split'])
    data_validation, labels_validation = data_train[idx_validation], labels_train[idx_validation]
    data_train, labels_train = data_train[idx_train], labels_train[idx_train]
    dataset_config['idx_train'] = idx_train
    dataset_config['idx_validation'] = idx_validation


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
# model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, dataset_train, dataset_validation)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if not debug :
    with open(path_traing_config, "w") as toml_file:
        train_config['name_training_run'] = ''
        toml.dump(train_config, toml_file)
