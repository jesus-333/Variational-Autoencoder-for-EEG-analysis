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
        if train_config['use_classifier']:
            x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predict_label = model(x)
        else:
            x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)
            true_label = None
            predict_label = None
        
        # Loss evaluation
        batch_train_loss = loss_function.compute_loss(x, x_r, 
                                                     mu_list, log_var_list, 
                                                     delta_mu_list, delta_log_var_list,
                                                     predict_label, true_label)
    
        # Backward/Optimization pass
        batch_train_loss[0].backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss[0] * x.shape[0]
        recon_loss += batch_train_loss[1] * x.shape[0]
        kl_loss    += batch_train_loss[2] * x.shape[0]
        if train_config['use_classifier']: clf_loss += batch_train_loss[4] * x.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)
    recon_loss = recon_loss / len(train_loader.sampler)
    kl_loss    = kl_loss / len(train_loader.sampler)
    if train_config['use_classifier']: clf_loss /= len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss']       = float(train_loss)
        log_dict['train_loss_recon'] = float(recon_loss)
        log_dict['train_kl_loss']    = float(kl_loss)

        if train_config['use_classifier']: log_dict['train_loss_clf'] = float(clf_loss)
        print("TRAIN LOSS")
        pprint.pprint(log_dict)
    
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config, log_dict = None):
    # Set the model in evaluation mode
    model.eval()

    # Variable to accumulate the loss
    validation_loss = 0
    recon_loss = 0
    kl_loss = 0
    clf_loss = 0

    for sample_data_batch, sample_label_batch in validation_loader:
        # Move data to training device
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        # Disable gradient tracking
        with torch.no_grad():
            # Forward pass
            if train_config['use_classifier']:
                x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predict_label = model(x)
            else:
                x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)
                true_label = None
                predict_label = None
            
            # Loss evaluation
            batch_validation_loss = loss_function.compute_loss(x, x_r,
                                                            mu_list, log_var_list,
                                                            delta_mu_list, delta_log_var_list,
                                                            predict_label, true_label)
            # Accumulate loss
            validation_loss += batch_validation_loss[0] * x.shape[0]
            recon_loss      += batch_validation_loss[1] * x.shape[0]
            kl_loss         += batch_validation_loss[2] * x.shape[0]
            if train_config['use_classifier']: clf_loss += batch_validation_loss[4] * x.shape[0]

    # Compute final loss
    validation_loss = validation_loss / len(validation_loader.sampler)
    recon_loss = recon_loss / len(validation_loader.sampler)
    kl_loss = kl_loss / len(validation_loader.sampler)
    if train_config['use_classifier']: clf_loss /= len(validation_loader.sampler)
    
    if log_dict is not None:
        log_dict['validation_loss'] = float(validation_loss)
        log_dict['validation_loss_recon'] = float(recon_loss)
        log_dict['validation_kl_loss'] = float(kl_loss)
        # for i in range(len(validation_loss[3])):
        #     kl_loss = validation_loss[3][i]
        #     log_dict['validation_loss_recon_{}'.format(i+1)] = validation_loss
        if train_config['use_classifier'] : log_dict['validation_loss_clf'] = float(clf_loss)
        print("VALIDATION LOSS")
        pprint.pprint(log_dict)
    
    return validation_loss
  
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 