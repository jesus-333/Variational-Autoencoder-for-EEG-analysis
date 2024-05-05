"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Training script (with wandb) for ChWi Version 1 (V1)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json

from library.dataset import dataset_time as ds_time, preprocess as pp
from library.training import wandb_training as wt

from library.config import config_training as ct
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameters to change inside the dictionary

# Subject of dataset 2a to use during training
subj_list = [3, 8, 9]

# Training parameters to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 80
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 2
project_name = "ChWi_Model"
name_training_run = "ChWi_v1_S3_S8_S9_training_run_1"
model_artifact_name = "ChWi_V1"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get dataset

# Get and preprocess d2a data. Preprocess is based on option in dataset_config
dataset_config = cd.get_moabb_dataset_config(subj_list)
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get tensor with data
train_data = train_dataset.data.squeeze()
validation_data = validation_dataset.data.squeeze()

# Create train and validation ChWi dataset
train_dataset = ds_time.EEG_Dataset_ChWi(train_data)
validation_dataset = ds_time.EEG_Dataset_ChWi(validation_data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model config

model_config_path = "json_config/ChWi_encoder.json"

with open(model_config_path, 'r') as j:
    model_config = json.loads(j.read())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get training config

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
train_config['measure_metrics_during_training'] = False
train_config['use_classifier'] = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Train the model

model = wt.train_wandb_V2('ChWi_autoencoder', train_config, model_config, train_dataset, validation_dataset)


plt.figure(figsize = (15, 10))
plt.plot(a[0].squeeze())
plt.plot(a_r[0].detach().squeeze())
plt.grid(True)
plt.show()