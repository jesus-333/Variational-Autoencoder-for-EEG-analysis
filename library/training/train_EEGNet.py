"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train EEGNet as classifier.
Works also with similar architecture like MBEEGNet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch

# Config files
from ..config import config_model as cm
from ..config import config_dataset as cd
from ..config import config_training as ct

from . import train_generic
    
"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Epochs function

def train_epoch(model, loss_function, optimizer, train_loader, train_config):
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    train_loss = 0

    for sample_data_batch, sample_label_batch in train_loader:
        # Move data to training device
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        predict_label = model(x)
        
        # Loss evaluation
        batch_train_loss = loss_function(predict_label, true_label)
    
        # Backward/Optimization pass
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss * x.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)
    
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config):
    # Set the model in evaluation mode
    model.eval()

    # Variable to accumulate the loss
    validation_loss = 0

    for sample_data_batch, sample_label_batch in validation_loader:
        # Move data to training device
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        # Disable gradient tracking
        with torch.no_grad():
            # Forward pass
            predict_label = model(x)

            # Loss evaluation
            batch_validation_loss = loss_function(predict_label, true_label)
            
            # Accumulate loss
            validation_loss += batch_validation_loss * x.shape[0]

    # Compute final loss
    validation_loss = validation_loss / len(validation_loader.sampler)
    
    return validation_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Main function
def main_stft():
    C = 22
    T = 512 

    dataset_config = cd.get_moabb_dataset_config([1])
    dataset_config['stft_parameters'] = cd.get_config_stft()

    train_config = ct.get_config_classifier()
    model_config = cm.get_config_EEGNet_stft_classifier(C, T, 22)

    train_generic.train_and_test_model('EEGNet', dataset_config, train_config, model_config)


if __name__ == "__main__":
    main_stft()
