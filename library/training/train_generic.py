
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with generic function to train a model in PyTorch. Contains the function that iterate through epochs, that is the same for each model.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch
import os
import sys
import wandb
import pprint

# Custom functions
from . import wandb_support
from .. import metrics
from . import loss_function
from ..dataset import preprocess as pp

# Config files
from ..config import config_model as cm
from ..config import config_dataset as cd
from ..config import config_training as ct
from .. import check_config

# Possible model to train
from ..model import EEGNet
from ..model import MBEEGNet
from ..model import vEEGNet
from ..model import hvEEGNet

# Training functions for specific model
from . import train_EEGNet
from . import train_vEEGNet
from . import train_hvEEGNet
    
"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_and_test_model(model_name, dataset_config, train_config, model_config, model_artifact = None):
    # torch.cuda.empty_cache()
    
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
    
    if dataset_config['use_stft_representation']:
        model_config['C'] = train_dataset[0][0].shape[0]
        model_config['T'] = train_dataset[0][0].shape[2]
        model_config['depth_first_layer'] = train_dataset[0][0].shape[0]
    
    # Create dataloader
    train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    loader_list             = [train_dataloader, validation_dataloader]
    test_dataloader         = torch.utils.data.DataLoader(test_dataset, batch_size = train_config['batch_size'], shuffle = True)
    
    # Create model
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model = get_untrained_model(model_name, model_config)
    model.to(train_config['device'])
    
    # Declare loss function
    loss_function = get_loss_function(model_name)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr = train_config['lr'], 
                                  weight_decay = train_config['optimizer_weight_decay']
                                  )

    # Setup lr scheduler
    if train_config['use_scheduler'] == True:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
    else:
        lr_scheduler = None

    # Create a folder (if not exist already) to store temporary file during training
    os.makedirs(train_config['path_to_save_model'], exist_ok = True)

    # (OPTIONAL)
    if train_config['wandb_training']: wandb.watch(model, log = "all", log_freq = train_config['log_freq'])

    # Train the model
    train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact)
    
    # TODO
    # test(model, test_dataloader, train_config)

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler = None, model_artifact = None):
    """
    Function with the training cycle
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Check train config
    check_config.check_train_config(train_config, model_artifact)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Get the dataloader
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    
    # Variable to track best losses
    best_loss_val = sys.maxsize # Best total loss for the validation data
    
    # (OPTIONAL) Dictionary used to saved information during training and load them on wandb
    log_dict = {}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    train_epoch_function, validation_epoch_function = get_train_and_validation_function(model)
    
    for epoch in range(train_config['epochs']):     
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (MANDATORY) Advance epoch, check validation loss and save the network

        # Advance epoch for train set (backward pass) and validation (no backward pass)
        # TODO add log dict to eegnet train/val function
        train_loss      = train_epoch_function(model, loss_function, optimizer, train_loader, train_config, log_dict)
        validation_loss = validation_epoch_function(model, loss_function, validation_loader, train_config, log_dict)
        
        # Save the new BEST model if a new minimum is reach for the validation loss
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))

        # Save the model after the epoch
        # N.b. When the variable epoch is n the model is trained for n + 1 epochs when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (OPTIONAL) Optional steps during the training

        # (OPTIONAL) Measure the various metrics
        if train_config['measure_metrics_during_training']:
            # Compute the various metrics
            train_metrics_dict = metrics.compute_metrics(model, train_loader, train_config['device'])    
            validation_metrics_dict = metrics.compute_metrics(model, validation_loader, train_config['device'])

        # (OPTIONAL) Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: 
            # Save the current learning rate if I load the data on wandb
            if train_config['wandb_training']: log_dict['learning_rate'] = optimizer.param_groups[0]['lr']

            # Update scheduler
            lr_scheduler.step()

        # (OPTIONAL) Print loss 
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print("\t Train loss        = {}".format(train_loss.detach().cpu().float()))
            print("\t Validation loss   = {}".format(validation_loss.detach().cpu().float()))

            if lr_scheduler is not None: print("\t Learning rate     = {}".format(optimizer.param_groups[0]['lr']))
            if train_config['measure_metrics_during_training']:
                print("\t Accuracy (TRAIN)  = {}".format(train_metrics_dict['accuracy']))
                print("\t Accuracy (VALID)  = {}".format(validation_metrics_dict['accuracy']))

        # (OPTIONAL) Log data on wandb
        if train_config['wandb_training']:
            # Update the log with the epoch losses
            log_dict['train_loss'] = train_loss
            log_dict['validation_loss'] = validation_loss
        
            # Save the metrics in the log 
            if train_config['measure_metrics_during_training']:
                wandb_support.update_log_dict_metrics(train_metrics_dict, log_dict, 'train')
                wandb_support.update_log_dict_metrics(validation_metrics_dict, log_dict, 'validation')
            
            # Add the model to the artifact
            if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
                wandb_support.add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))
            
            wandb.log(log_dict)
        
        # End training cycle
    
    # Save the model with the best loss on validation set
    if train_config['wandb_training']:
        wandb_support.add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))

def test(model, test_loader, config):
    print("Metrics at the end of the training (END)")
    metrics_dict = metrics.compute_metrics(model, test_loader, 'cpu')
    pprint.pprint(metrics_dict)
    
    print("Metrics at the minimum of validation loss (BEST)")
    model.load_state_dict(torch.load(config['path_to_save_model'] + '/model_BEST.pth'))
    metrics_dict = metrics.compute_metrics(model, test_loader, 'cpu')
    pprint.pprint(metrics_dict)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_untrained_model(model_name : str, model_config : dict):
    if model_name == 'EEGNet':
        return EEGNet.EEGNet_Classifier(model_config)
    elif model_name == 'MBEEGNet':
        return MBEEGNet.MBEEGNet_Classifier(model_config)
    elif model_name == 'vEEGNet':
        return vEEGNet.vEEGNet(model_config)
    elif model_name == 'hvEEGNet_shallow':
        return hvEEGNet.hvEEGNet_shallow(model_config)
    else:
        raise ValueError("Type of the model not recognized")

def get_loss_function(model_name, config = None):
    if model_name == 'EEGNet' or model_name == 'MBEEGNet':
        return torch.nn.NLLLoss()
    elif model_name == 'vEEGNet':
        return loss_function.vEEGNet_loss 
    elif model_name == 'hvEEGNet_shallow':
        return loss_function.hvEEGNet_loss
    else:
        raise ValueError("Type of the model not recognized")

def get_train_and_validation_function(model):
    if 'EEGNet.EEGNet' in str(type(model)):
        return train_EEGNet.train_epoch, train_EEGNet.validation_epoch
    elif 'MBEEGNet.MBEEGNet' in str(type(model)):
        return train_EEGNet.train_epoch, train_EEGNet.validation_epoch
    elif 'vEEGNet.vEEGNet' in str(type(model)):
        return train_vEEGNet.train_epoch, train_vEEGNet.validation_epoch
    elif 'hvEEGNet.hvEEGNet_shallow' in str(type(model)):
        return train_hvEEGNet.train_epoch, train_hvEEGNet.validation_epoch
    else:
        raise ValueError("Type of the model not recognized")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
