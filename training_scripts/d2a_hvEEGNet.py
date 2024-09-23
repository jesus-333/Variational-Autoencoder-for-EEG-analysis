"""
Train hvEEGNet with the data from dataset 2a of BCI Competition IV

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import toml
import numpy as np

from library import check_config
from library.dataset import preprocess as pp
from library.training import wandb_training as wt


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj = 7

# Path of config files
path_dataset_config = 'training_scripts/config/d2a/dataset.toml'
path_model_config   = 'training_scripts/config/d2a/model.toml'
path_traing_config  = 'training_scripts/config/d2a/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_config = toml.load(path_dataset_config)
dataset_config['subjects_list'] = [subj]
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get model config
model_config = toml.load(path_model_config)
check_config.check_model_config_hvEEGNet(model_config)

# Get training config
train_config = toml.load(path_traing_config)
check_config.check_train_config(train_config)

# Train the model
# model = wt.train_wandb_V2('hvEEGNet_shallow', train_config, model_config, train_dataset, validation_dataset)
