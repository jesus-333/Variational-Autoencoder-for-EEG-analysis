"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train vEEGNet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch
import wandb
import os
import sys

# Custom functions
import wandb_support
import metrics
import dataset
import MBEEGNet

# Config files
import config_model as cm
import config_dataset as cd
import config_training as ct
import loss_function as lf
    
"""
%load_ext autoreload
%autoreload 2

import train_vEEGNet 
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_and_test_model(dataset_config, train_config, model_config, model_artifact = None):
    # Get the training data
    train_dataset, validation_dataset = dataset.get_train_data(dataset_config)

    # Create dataloader
    train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    loader_list             = [train_dataloader, validation_dataloader]

    # Get EEG Channels and number of samples
    C = train_dataset[0][0].shape[1] 
    T = train_dataset[0][0].shape[2]

    # Create model
    model = MBEEGNet.MBEEGNet_Classifier(model_config)
    model.to(train_config['device'])

def train():
    pass

def test():
    pass
