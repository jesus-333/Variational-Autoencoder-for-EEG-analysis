"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train MBEEGNet as classifier
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
import config_model

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_and_test_model(dataset_config, train_config, model_artifact = None):
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
    model_config = config_model.get_config_MBEEGNet_classifier(C, T, 4)
    model = MBEEGNet.MBEEGNet_Classifier(model_config)
    model.to(train_config['device'])
    
    # Declare loss function
    loss_function = torch.nn.NLLLoss()

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
    # test(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact)

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler = None, model_artifact = None):
    """
    Function with the training cycle
    """

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Check train config
    check_train_config(train_config, model_artifact)

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
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (MANDATORY) Advance epoch, check validation loss and save the network

        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss      = train_epoch(model, loss_function, optimizer, train_loader, train_config)
        validation_loss = validation_epoch(model, loss_function, validation_loader, train_config)
        
        # Save the new BEST model if a new minimum is reach for the validation loss
        if validation_loss < best_loss_val:
            best_loss_val = validation_loss
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))

        # Save the model after the epoch
        # N.b. When the variable epoch is n the model is trained for n + 1 epochs when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # (OPTIONAL) Optional step during the training

        # (OPTIONAL) Measure the various metrics
        if train_config['measure_metrics_during_training']:
            # Compute the various metrics
            train_metrics_list = metrics.compute_metrics(model, train_loader, train_config['device'])    
            validation_metrics_list = metrics.compute_metrics(model, validation_loader, train_config['device'])
            

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
                print("\t Accuracy (TRAIN)  = {}".format(train_metrics_list[0]))
                print("\t Accuracy (VALID)  = {}".format(validation_metrics_list[0]))

        # (OPTIONAL) Log data on wandb
        if train_config['wandb_training']:
            # Update the log with the epoch losses
            log_dict['train_loss'] = train_loss
            log_dict['validation_loss'] = validation_loss
        
            # Save the metrics in the log 
            if train_config['measure_metrics_during_training']:
                wandb_support.update_log_dict_metrics(train_metrics_list, log_dict, 'train')
                wandb_support.update_log_dict_metrics(validation_metrics_list, log_dict, 'validation')
            
            # Add the model to the artifact
            if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
                wandb_support.add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))
            
            wandb.log(log_dict)
        
        # End training cycle
    
    # Save the model with the best loss on validation set
    if train_config['wandb_training']:
        wandb_support.add_file_to_artifact(model_artifact, '{}/{}'.format(train_config['path_to_save_model'], 'model_BEST.pth'))


def test():
    pass

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
#%% Other function

def check_train_config(train_config : dict, model_artifact = None):
    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
    
    # Check if wandb is used during training
    if 'wandb_training' not in train_config: train_config['wandb_training'] = False
    if train_config['wandb_training'] == True and model_artifact is None: raise ValueError("If you want to train the model and load the data on wandb you must also pass an artifact to save the network") 
    
    # Path where save the network during training
    if 'path_to_save_model' not in train_config:
        print("path_to_save_model not found. Used current directory")
        train_config['path_to_save_model'] = "."
    
    if 'measure_metrics_during_training' not in train_config:
        print("measure_metrics_during_training not found. Set to False")
        train_config['measure_metrics_during_training'] = False

    if 'print_var' not in train_config:
        print("print_var not found. Set to True")
        train_config['print_var'] = True


