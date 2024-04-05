""" 
Example of training script for hvEEGNet with dataset 2a of BCI Competition IV and use of wandb to log results
The dataset is automatically downloaded through the library.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.training import wandb_training as wt

from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameter to change inside the dictionary

# Subject of dataset 2a to use during training
subj_list = [3]

# Training config to change (for more info check the function get_config_hierarchical_vEEGNet_training)
epochs = 2
path_to_save_model = 'model_weights_backup'
epoch_to_save_model = 1
project_name = "Example_project"               # Name of wandb project
model_artifact_name = "temporary_artifacts"    # Name of the artifact used to save the model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get dataset config
dataset_config = cd.get_moabb_dataset_config(subj_list)

# Get model config
C = 22  # Number of EEG channels of dataset 2a
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf) # Compute number of time samples
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

# Update training config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model

# Update training config (wandb)
train_config['project_name'] = project_name
train_config['model_artifact_name'] = model_artifact_name

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# Train the model
model = wt.train_wandb_V1('hvEEGNet_shallow', dataset_config, train_config, model_config)
