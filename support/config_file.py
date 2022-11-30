"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function with the config for the training and to get model and data
"""

"""
TO EXECUTE
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, 'support')
import wandb_training
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import torch
from torch.utils.data import DataLoader

from VAE_EEGNet import EEGFramework
from support_datasets import PytorchDatasetEEGSingleSubject, PytorchDatasetEEGMergeSubject

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Config dictionary

def get_dataset_config():
    dataset_config = dict(
        train_path = 'Dataset/D2A/v2_raw_128/Train/',
        test_path = 'Dataset/D2A/v2_raw_128/Test/',
        merge_list = [1,2,3,4,5,6,7,8,9],
        normalize_trials = False,
        percentage_split = 0.9,
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )

    return dataset_config

def get_wandb_config(run_name):
    wandb_config = dict(
        project_name = "VAE_EEG",
        run_name = run_name
    )

    return wandb_config

def get_train_config():
    train_config = dict(
        model_artifact_name = "vEEGNet",    # Name of the artifact used to save the models
        # Training settings
        batch_size = 20,                    
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 200,                       # Number of epochs to train the model
        use_scheduler = False,               # Use the lr scheduler
        lr_decay_rate = 0.99,               # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-5,      # Weight decay of the optimizer
        use_shifted_VAE_loss = False,
        L2_loss_type = 0,
        # Loss multiplication factor
        alpha = 0.1,                          # Reconstruction
        beta = 1,                           # KL
        gamma = 1,                          # Discrimination
        # Support stuff (device, log frequency etc)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        log_freq = 1,
        epoch_to_save_model = 5,
        measure_accuracy_during_training = True,
        repetition = 1,                     # Number of time to repeat the training 
        print_var = True,
        debug = False,                       # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )
    

    return train_config


def get_metrics_config():
    """
    Config to download the metrics from wandb 
    """

    metrics_config = dict(
        metrics_artifact_name = "jesus_333/VAE_EEG/Metrics",
        version_list = [5, 6, 7],
        metric_names = ["accuracy", "cohen_kappa", "sensitivity", "specificity", "f1"]
    )

    return metrics_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model and data function

def get_train_data(config):
    """
    Return the training data of the D2A dataset  
    """
    full_dataset = PytorchDatasetEEGMergeSubject(config['train_path'], idx_list = config['merge_list'], 
                                                 normalize_trials = config['normalize_trials'], 
                                                 optimize_memory = False, device = config['device'])
    
    train_dataset, validation_dataset = split_dataset(full_dataset, config['percentage_split'])

    return train_dataset, validation_dataset

def get_test_data(config, return_dataloader = True, batch_size = 32):
    idx_list = [1,2,3,4,5,6,7,8,9]
    data_list = []

    for idx in idx_list:
        path = config['test_path'] + '{}/'.format(idx)
        dataset_subject = PytorchDatasetEEGSingleSubject(path, normalize_trials = config['normalize_trials'])
            
        if return_dataloader: data_list.append(DataLoader(dataset_subject, batch_size = batch_size, shuffle = True))
        else: data_list.append(dataset_subject)

    return data_list

def split_dataset(full_dataset, percentage_split):
    """
    Split a dataset in 2 
    """

    size_train = int(len(full_dataset) * percentage_split) 
    size_val = len(full_dataset) - size_train
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [size_train, size_val])
    
    return train_dataset, validation_dataset

def get_model(C, T, hidden_space_dimension):
        eeg_framework = EEGFramework(C = C, T = T, hidden_space_dimension = hidden_space_dimension, 
                                     use_reparametrization_for_classification = False, 
                                     print_var = True, tracking_input_dimension = False)

        return eeg_framework
