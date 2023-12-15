"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Train function of the hierarchical vEEGnet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch
import pprint

# Config files
from ..config import config_model as cm
from ..config import config_dataset as cd
from ..config import config_training as ct
    
"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Epochs function

def train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    clf_loss = 0

    for sample_data_batch, sample_label_batch in train_loader:
        # Move data to training device
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        predict_label = model(x)
        
        # Loss evaluation
        batch_train_loss = loss_function.compute_loss(predict_label, true_label)
    
        # Backward/Optimization pass
        batch_train_loss.backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss * x.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss'] = float(train_loss)
        print("TRAIN LOSS")
        pprint.pprint(log_dict)
    
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config, log_dict = None):
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
            batch_validation_loss = loss_function.compute_loss(predict_label, true_label)

            # Accumulate loss
            validation_loss += batch_validation_loss * x.shape[0]

    # Compute final loss
    validation_loss = validation_loss / len(validation_loader.sampler)
    
    if log_dict is not None:
        log_dict['validation_loss'] = float(validation_loss)
        print("VALIDATION LOSS")
        pprint.pprint(log_dict)
    
    return validation_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    pass

if __name__ == '__main__':
    main()
