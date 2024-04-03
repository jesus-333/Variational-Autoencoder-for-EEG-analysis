"""
Example of training script for hvEEGNet with external data and use of wandb to log results

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np

from library.dataset import dataset_time as ds_time
from library.training import wandb_training as wt

from library.config import config_training as ct
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameter to change inside the dictionary

type_decoder = 0
parameters_map_type = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create synthetic data
train_data = np.random.rand((100, 1 , 3, 1000))
validation_data = np.random.rand((100, 1 , 3, 1000))

# Create channel lists
ch_list = ['C3', 'C5', 'C6']

# Create synthetic label
train_label = np.random.randint(0, 4, train_data.shape[0])
validation_label = np.random.randint(0, 4, validation_data.shape[0])

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get training config
train_config = ct.get_config_vEEGNet_training()

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# Create train and validation dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, ch_list)

# Train the model
model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset)
