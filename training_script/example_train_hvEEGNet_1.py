""" 
Example of training script for hvEEGNet with dataset 2a of BCI Competition IV and use of wandb to log results

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.training import wandb_training as wt
from library.dataset import preprocess as pp

from library.config import config_training as ct
from library.config import config_model as cm
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameter to change inside the dictionary

subj_list = [1, 2, 3, 4]

type_decoder = 0
parameters_map_type = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get dataset config
dataset_config = cd.get_moabb_dataset_config(subj_list)

# Get training config
train_config = ct.get_config_vEEGNet_training()

# Get model config
C = 22
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf)
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# Get the data
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Train the model
model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset)
