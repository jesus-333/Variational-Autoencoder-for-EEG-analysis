"""
Example of training script for hvEEGNet with external data and use of wandb to log results

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch

from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic

from library.config import config_training as ct
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameter to change inside the dictionary

type_decoder = 0
parameters_map_type = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data and train config

# Create synthetic data
train_data = np.random.rand((100, 1 , 3, 1000))
validation_data = np.random.rand((100, 1 , 3, 1000))

# Create channel lists
ch_list = ['C3', 'C5', 'C6']

# Create synthetic label
train_label = np.random.randint(0, 4, train_data.shape[0])
validation_label = np.random.randint(0, 4, validation_data.shape[0])

# Create train and validation dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, ch_list)

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']


# hvEEGNet creation
model = hvEEGNet.hvEEGNet_shallow(model_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, Loss function, optimizer and lr_scheduler

# Create dataloader
train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
loader_list             = [train_dataloader, validation_dataloader]

# Declare loss function
# This method return the PyTorch loss function required by the training function.
# The loss function for hvEEGNet is not directy implemented in PyTorch since it is a combination of different losses. So I have to create my own function to combine all the components.
loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', train_config = train_config)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = train_config['lr'],
                              weight_decay = train_config['optimizer_weight_decay']
                              )

# (OPTIONAL) Setup lr scheduler
if train_config['use_scheduler'] :
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
else:
    lr_scheduler = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from library.training import train_generic

model = train_generic.train(model, loss_function, optimizer,
                            loader_list, train_config, lr_scheduler, model_artifact = None)


