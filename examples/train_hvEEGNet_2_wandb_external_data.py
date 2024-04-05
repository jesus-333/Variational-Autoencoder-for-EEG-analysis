"""
Example of training script for hvEEGNet with external data (i.e. data that you provide) and use of wandb to log results.
For this example I create synthetic data and lables.

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
# Specific parameters to change inside the dictionary

# Training parameters to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 2
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 1
project_name = "Example_project"                # Name of wandb project
name_training_run = "first_test_wandb"          # Name of the training run
model_artifact_name = "temporary_artifacts"     # Name of the artifact used to save the model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create synthetic data
train_data = np.random.rand(20, 1 , 3, 1000)
validation_data = np.random.rand(20, 1 , 3, 1000)

# Create channel lists
ch_list = ['C3', 'C5', 'C6']

# Create synthetic label
train_label = np.random.randint(0, 4, train_data.shape[0])
validation_label = np.random.randint(0, 4, validation_data.shape[0])

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Get training config
train_config = ct.get_config_vEEGNet_training()

# Update training config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model

# Update training config (wandb)
train_config['project_name'] = project_name
train_config['name_training_run'] = name_training_run
train_config['model_artifact_name'] = model_artifact_name

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# Create train and validation dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, ch_list)

# Train the model
model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset)
