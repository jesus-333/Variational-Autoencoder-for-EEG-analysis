"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the model with the wandb framework
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import sys
import os
import wandb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import compute_metrics
from support_training import VAE_and_classifier_loss
import config_file as cf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Principal function

def train_and_test_model_wandb(hidden_space = 16):
    # Get the various config
    dataset_config = cf.get_dataset_config()
    train_config = cf.get_train_config()

    # Get the training data
    train_dataset, validation_dataset = cf.get_train_data(dataset_config)
    
    # Create dataloader
    train_dataloader        = DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    validation_dataloader   = DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    loader_list             = [train_dataloader, validation_dataloader]
    
    # Get test data
    test_loader_list = cf.get_test_data(dataset_config, return_dataloader = True, batch_size = train_config['batch_size'])

    # Variables
    C = train_dataset[0][0].shape[1] # Used for model creation
    T = train_dataset[0][0].shape[2] # Used for model creation

    # Train the model 
    for rep in range(train_config['repetition']):
        # Create the model 
        eeg_framework = cf.get_model(C, T, hidden_space)

        # Train the model
        wandb_config = cf.get_wandb_config('train_VAE_EEG_{}'.format(rep))
        model = train_model_wandb(eeg_framework, loader_list, train_config, wandb_config)
    
        # Test the model
        wandb_config = cf.get_wandb_config('test_VAE_EEG_{}'.format(rep))
        df_metrics_list = test_model_wandb(eeg_framework, test_loader_list, train_config, wandb_config)
    
    return model, df_metrics_list

def train_model_wandb(model, loader_list, train_config, wandb_config):
    # Create a folder (if not exist already) to store temporary file during training
    os.makedirs('TMP_File', exist_ok = True)

    with wandb.init(project = wandb_config['project_name'], job_type = "train", config = train_config, name = wandb_config['run_name']) as run:
        train_config = wandb.config

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = train_config['lr'], 
                                      weight_decay = train_config['optimizer_weight_decay'])

        # Setup lr scheduler
        if train_config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
        else:
            lr_scheduler = None
            
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)
        
        # Print the training device
        if train_config['print_var']: print("Model trained on: {}".format(train_config['device']))

        # Train model
        wandb.watch(model, log = "all", log_freq = train_config['log_freq'])
        model.to(train_config['device'])
        train_cycle(model, optimizer, loader_list, model_artifact, train_config, lr_scheduler)

        # Save model after training
        add_model_to_artifact(model, model_artifact, "TMP_File/model_END.pth")
        run.log_artifact(model_artifact)
        
    return model


def test_model_wandb(model, test_loader_list, config, wandb_config):
    with wandb.init(project = wandb_config['project_name'], job_type = "test", config = config, name = wandb_config['run_name']) as run:
        # Variable to save results
        metrics_END = np.zeros((9, 5))
        metrics_BEST_TOT = np.zeros((9, 5))
        metrics_BEST_CLF = np.zeros((9, 5))

        # Compute metrics at the end of the train
        for i in range(len(test_loader_list)):
            loader = test_loader_list[i]

            metrics_END[i, :] = np.asarray(compute_metrics(model, loader, config['device']))
            
        # Compute metrics when the model reach the best loss
        for i in range(len(test_loader_list)):
            loader = test_loader_list[i]
            
            # Best total loss
            model.load_state_dict(torch.load("TMP_File/model_BEST_TOTAL.pth"))
            metrics_BEST_TOT[i, :] = np.asarray(compute_metrics(model, loader, config['device']))

            # Best classifier loss
            model.load_state_dict(torch.load("TMP_File/model_BEST_CLF.pth"))
            metrics_BEST_CLF[i, :] = np.asarray(compute_metrics(model, loader, config['device']))
        
        # Save the matrix in pandas dataframe
        columns_names = ["accuracy", "cohen_kappa", "sensitivity", "specificity", "f1"]
        df_END = pd.DataFrame(data = metrics_END, index = [1,2,3,4,5,6,7,8,9], columns = columns_names)
        df_BEST_TOT = pd.DataFrame(data = metrics_BEST_TOT, index = [1,2,3,4,5,6,7,8,9], columns = columns_names)
        df_BEST_VAL = pd.DataFrame(data = metrics_BEST_CLF, index = [1,2,3,4,5,6,7,8,9], columns = columns_names)

        # Save dataframe to CSV
        df_END.to_csv('TMP_File/metrics_END.csv')
        df_BEST_TOT.to_csv('TMP_File/metrics_BEST_TOT.csv')
        df_BEST_VAL.to_csv('TMP_File/metrics_BEST_CLF.csv')
        
        # Create wandb artifact and save results
        metrics_artifact = wandb.Artifact('Metrics', type = "metrics")
        add_metrics_to_artifacts('TMP_File/metrics_END.csv', metrics_artifact)
        add_metrics_to_artifacts('TMP_File/metrics_BEST_TOT.csv', metrics_artifact)
        add_metrics_to_artifacts('TMP_File/metrics_BEST_CLF.csv', metrics_artifact)

        run.log_artifact(metrics_artifact)

        return df_END, df_BEST_TOT, df_BEST_VAL

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train cycle function

def train_cycle(model, optimizer, loader_list, model_artifact, train_config, lr_scheduler = None):
    """
    Function with the training cycle
    """

    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
    
    # Variable used during the traing
    train_loader = loader_list[0]
    validation_loader = loader_list[1]
    log_dict = {}
    best_loss_val_tot = sys.maxsize # Best total loss
    best_loss_val_clf = sys.maxsize # Best classifier loss
    
    for epoch in range(train_config['epochs']):
        # Save lr
        if train_config['use_scheduler']:
            log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss_list      = advance_epoch(model, optimizer, train_loader, train_config, True)
        validation_loss_list = advance_epoch(model, optimizer, validation_loader, train_config, False)
        
        # Save the new BEST model if a new minimum is reach for the validation loss (TOTAL)
        if validation_loss_list[0] < best_loss_val_tot:
            best_loss_val_tot = validation_loss_list[0]
            torch.save(model.state_dict(), 'TMP_File/model_BEST_TOTAL.pth')

        # Save the new BEST model if a new minimum is reach for the validation loss (CLF)
        if validation_loss_list[3] < best_loss_val_clf:
            best_loss_val_clf = validation_loss_list[3]
            torch.save(model.state_dict(), 'TMP_File/model_BEST_CLF.pth')
        
        # Update the log with the epoch loss
        update_log_dict_loss(train_loss_list, log_dict, 'train')
        update_log_dict_loss(validation_loss_list, log_dict, 'validation')

        # Measure the various metrics
        if train_config['measure_metrics_during_training']:
            # Compute the various metrics
            train_metrics_list = compute_metrics(model, train_loader, train_config['device'])    
            validation_metrics_list = compute_metrics(model, validation_loader, train_config['device'])
            
            # Save the metrics in the log
            update_log_dict_metrics(train_metrics_list, log_dict, 'train')
            update_log_dict_metrics(validation_metrics_list, log_dict, 'validation')

        # Save the model after the epoch
        # N.b. When the variable epoch is 0 the model is trained for an epoch when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        # Print loss 
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print(get_loss_string(log_dict))

        # End training cycle
    
    # Save the model with the best loss on validation set
    model_artifact.add_file('TMP_File/model_BEST_TOTAL.pth')
    model_artifact.add_file('TMP_File/model_BEST_CLF.pth')
    wandb.save()


def advance_epoch(model, optimizer, loader, train_config, is_train):
    """
    Function to advance a single epoch of the model
    """
    
    if is_train: model.train()
    else: model.eval()
    
    tot_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    tot_discriminator_loss = 0

    for sample_data_batch, sample_label_batch in loader:
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        if is_train:
            # Zeros past gradients
            optimizer.zero_grad()
            
            # Networks forward pass
            x_r, mu, log_var, predict_label = model(x)
            
            # Loss evaluation
            loss_list = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, 
                                                train_config['use_shifted_VAE_loss'], 
                                                train_config['alpha'], train_config['beta'], train_config['gamma'], 
                                                train_config['L2_loss_type'])
            
            total_loss = loss_list[0]
            # total_loss, recon_loss, kl_loss, discriminator_loss
        
            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                # Forward pass
                x_r, mu, log_var, predict_label = model(x)
            
                # Loss evaluation
                loss_list = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, 
                                                    train_config['use_shifted_VAE_loss'], 
                                                    train_config['alpha'], train_config['beta'], train_config['gamma'], 
                                                    train_config['L2_loss_type'])
                total_loss = loss_list[0]    

        # cccumulate the  loss
        tot_loss += total_loss * x.shape[0]
        tot_recon_loss += loss_list[1] * x.shape[0]
        tot_kl_loss += loss_list[2] * x.shape[0]
        tot_discriminator_loss += loss_list[3] * x.shape[0]

    # End training cycle

    # Compute final loss
    tot_loss = tot_loss / len(loader.sampler)
    tot_recon_loss = tot_recon_loss / len(loader.sampler)
    tot_kl_loss = tot_kl_loss / len(loader.sampler)
    tot_discriminator_loss = tot_discriminator_loss / len(loader.sampler)
    
    return [tot_loss, tot_recon_loss, tot_kl_loss, tot_discriminator_loss]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other function

def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)

def add_metrics_to_artifacts(metrics_file_name, artifact):
    artifact.add_file(metrics_file_name)
    wandb.save(metrics_file_name)

def update_log_dict_loss(loss_list, log_dict, label):
    tot_loss = loss_list[0]
    recon_loss = loss_list[1]
    kl_loss = loss_list[2]
    discriminator_loss = loss_list[3]

    log_dict["tot_loss_{}".format(label)] = tot_loss
    log_dict["recon_loss_{}".format(label)] = recon_loss
    log_dict["kl_loss_{}".format(label)] = kl_loss
    log_dict["discriminator_loss_{}".format(label)] = discriminator_loss


def update_log_dict_metrics(metrics_list, log_dict, label):
    # return accuracy, cohen_kappa, sensitivity, specificity, f1
    accuracy = metrics_list[0]
    cohen_kappa = metrics_list[1]
    sensitivity = metrics_list[2]
    specificity = metrics_list[3]
    f1 = metrics_list[4]

    log_dict['accuracy_{}'.format(label)] = accuracy
    log_dict['cohen_kappa_{}'.format(label)] = cohen_kappa
    log_dict['sensitivity_{}'.format(label)] = sensitivity
    log_dict['specificity_{}'.format(label)] = specificity
    log_dict['f1_{}'.format(label)] = f1


def get_loss_string(log_dict):
    tmp_loss = ""
    
    tmp_loss += "\t(TRAIN) Tot   Loss:\t" + str(float(log_dict['tot_loss_train'])) + "\n"
    tmp_loss += "\t(TRAIN) Recon Loss:\t" + str(float(log_dict['recon_loss_train'])) + "\n"
    tmp_loss += "\t(TRAIN) KL    Loss:\t" + str(float(log_dict['kl_loss_train'])) + "\n"
    tmp_loss += "\t(TRAIN) Disc  Loss:\t" + str(float(log_dict['discriminator_loss_train'])) + "\n\n"

    tmp_loss += "\t(VALID) Tot   Loss:\t" + str(float(log_dict['tot_loss_validation'])) + "\n"
    tmp_loss += "\t(VALID) Recon Loss:\t" + str(float(log_dict['recon_loss_validation'])) + "\n"
    tmp_loss += "\t(VALID) KL    Loss:\t" + str(float(log_dict['kl_loss_validation'])) + "\n"
    tmp_loss += "\t(VALID) Disc  Loss:\t" + str(float(log_dict['discriminator_loss_validation'])) + "\n"

    return tmp_loss
