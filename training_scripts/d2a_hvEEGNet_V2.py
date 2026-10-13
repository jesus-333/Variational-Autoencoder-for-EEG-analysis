"""
Train hvEEGNet with the data from dataset 2a of BCI Competition IV

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import toml

from library import check_config
from library.dataset import preprocess as pp
from library.training import wandb_training as wt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj = 3

# Path of config files
path_dataset_config = 'training_scripts/config/d2a_hvEEGNet_V2/dataset.toml'
path_model_config   = 'training_scripts/config/d2a_hvEEGNet_V2/model.toml'
path_traing_config  = 'training_scripts/config/d2a_hvEEGNet_V2/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dataset_config = toml.load(path_dataset_config)
dataset_config['subjects_list'] = [subj]
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# This is temporary. For now I use the automatic moabb segmenetation because it also automatically applied the resampling
# For these experiments I want to resample the signal to 128Hz ---> 512 samples for 4 seconds of signal
# I do not know why but the resampling results in 513 samples instead of 512, so (as a temporary solution) I take only the first 512 samples
train_dataset.data = train_dataset.data[:,:,:,0:512]
validation_dataset.data = validation_dataset.data[:,:,:,0:512]
test_dataset.data = test_dataset.data[:,:,:,0:512]

# Get model config
model_config = toml.load(path_model_config)
check_config.check_model_config_hvEEGNet(model_config)

# Get training config
train_config = toml.load(path_traing_config)
check_config.check_train_config(train_config)

# Train the model
model = wt.train_wandb_V2('hvEEGNet_V2', train_config, model_config, train_dataset, validation_dataset)
