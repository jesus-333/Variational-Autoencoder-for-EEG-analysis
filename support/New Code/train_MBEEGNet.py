"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train MBEEGNet as classifier
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

import sys
import wandb

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_cycle(model, optimizer, loader_list, train_config, lr_scheduler = None, model_artifact = None):
    """
    Function with the training cycle
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Check to config

    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
    
    if 'wandb_training' not in train_config: train_config['wandb_training'] = False
    if train_config['wandb_training'] == True and model_artifact is None: raise ValueError("If you want to train the model and load the data on wandb you must also pass an artifact to save the network") 
    
    if 'path_to_save_model' not in train_config:
        print("path_to_save_model not set. Used current directory")
        train_config['path_to_save_model'] = "."

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get the dataloader
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    
    # Variable to track best losses
    best_loss_val = sys.maxsize # Best total loss for the validation data
    
    # (OPTIONAL) Dictionary used to saved information during training and load them on wandb
    log_dict = {}
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    for epoch in range(train_config['epochs']):
        
        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss      = advance_epoch(model, optimizer, train_loader, train_config, True)
        validation_loss = advance_epoch(model, optimizer, validation_loader, train_config, False)
        
        # Save the new BEST model if a new minimum is reach for the validation loss
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            save_model(model, train_config['path_to_save_model'], 'model_BEST_VAL.pth')
        

        # Measure the various metrics
        if train_config['measure_metrics_during_training']:
            # Compute the various metrics
            train_metrics_list = compute_metrics(model, train_loader, train_config['device'])    
            validation_metrics_list = compute_metrics(model, validation_loader, train_config['device'])
            
            # Save the metrics in the log
            update_log_dict_metrics(train_metrics_list, log_dict, 'train')
            update_log_dict_metrics(validation_metrics_list, log_dict, 'validation')

        # Save the model after the epoch
        # N.b. When the variable epoch is n the model is trained for n+1 epochs when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: 
            # Save the current learning rate if I load the data on wandb
            if model_artifact is not None: log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
        
        # Print loss 
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print(get_loss_string(log_dict))

        # Log data on wandb
        if model_artifact is not None:
            # Update the log with the epoch loss
            update_log_dict_loss(train_loss_list, log_dict, 'train')
            update_log_dict_loss(validation_loss_list, log_dict, 'validation')
        
            wandb.log(log_dict)
        
        # End training cycle
    
    # Save the model with the best loss on validation set
    model_artifact.add_file('TMP_File/model_BEST_TOTAL.pth')
    model_artifact.add_file('TMP_File/model_BEST_CLF.pth')
    wandb.save()


def train_epoch():
    pass

def test_epoch():
    pass


def save_model(model, path_to_save_model : str, model_name : str):
    torch.save(model.state_dict(), '{}/{}'.format(path_to_save_model, model_name))
