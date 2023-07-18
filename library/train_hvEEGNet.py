"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Train function of the hierarchical vEEGnet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch

# Config files
from .config import config_model as cm
from .config import config_dataset as cd
from .config import config_training as ct
    
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
        x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)
        
        # Loss evaluation
        batch_train_loss = loss_function(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)
    
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
            x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)
            
            # Loss evaluation
            batch_validation_loss = loss_function(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)
            
            # Accumulate loss
            validation_loss += batch_validation_loss * x.shape[0]

    # Compute final loss
    validation_loss = validation_loss / len(validation_loader.sampler)
    
    return validation_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    pass

if __name__ == '__main__':
    main()
