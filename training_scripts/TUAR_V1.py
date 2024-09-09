"""
Train hvEEGNet with the data from TUAR dataset
For this example I create synthetic data and lables.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.dataset import preprocess as pp
from library.training import wandb_training as wt

from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameters to change inside the dictionary

path_file = 'data/TUAR_dataset/aaaaaaju_s007_t000.edf'
force_download = False

# Training parameters to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 2
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 1
project_name = "TUAR_train_hvEEGNet"                # Name of wandb project
name_training_run = "TUAR"          # Name of the training run
model_artifact_name = "hvEEGNet_TUAR"     # Name of the artifact used to save the model

device = 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get dataset config
dataset_config = cd.get_TUAR_dataset_config(path_file, force_download)

# Get dataset
train_dataset, validation_dataset, test_dataset = pp.get_dataset_TUAR(dataset_config)

# Get number of channels and length of time samples
C = train_dataset.data.shape[2]
T = train_dataset.data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)
model_config['use_classifier'] = False

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

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

train_config['device'] = device

# Train the model
model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset)
