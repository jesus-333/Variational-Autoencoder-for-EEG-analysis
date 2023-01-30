"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain function used to config the wandb sweep and during the sweep train
"""

"""
TO EXECUTE
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0, 'support')
import config_file as cf
import wandb_sweep
import wandb
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import wandb
import torch
from torch.utils.data import DataLoader

import numpy as np
import config_file as cf
from wandb_training import train_cycle, add_model_to_artifact
import moabb_dataset as md

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train function

def train_sweep(config = None):
    # Get other config not included in the ones used for the sweep
    # dataset_config = cf.get_dataset_config()
    dataset_config = cf.get_moabb_dataset_config()
    train_config = cf.get_train_config()
    
    #TODO REMOVE
    train_config['lr'] = np.random.choice([1e-3, 7 * 1e-4, 3 * 1e-4])
    # train_config['batch_size'] = np.random.choice([int(15), int(30), int(45)])
    
    notes = "Falso sweep con parametri già settati. Inoltre classifico solo la media"

    with wandb.init(project = "VAE_EEG", job_type = "train", config = config, notes = notes) as run:
        # Config from the sweep
        config = wandb.config
        print("Start Sweep")

        # "Correct" dictionaries with the parameters from the sweep
        correct_dataset_config(config, dataset_config)
        correct_train_config(config, train_config)
        print("Update config with sweep parameters")
        
        train_config['alpha'] = 1
        train_config['beta'] = 1
        train_config['gamma'] = 1
        print(train_config)
        
        # Get the training data
        # train_dataset, validation_dataset = cf.get_train_data(dataset_config)
        train_dataset, validation_dataset = md.get_train_data(dataset_config)
        print("Dataset created")
        print("dataset_config['normalize_trials']: ", dataset_config['normalize_trials'])

        # Create dataloader
        train_dataloader        = DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        validation_dataloader   = DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
        loader_list             = [train_dataloader, validation_dataloader]
        print("Dataloader created")
        
        # Get test data
        # test_loader_list = cf.get_subject_data(dataset_config, 'test', return_dataloader = True, batch_size = train_config['batch_size'])

        # Variables
        C = train_dataset[0][0].shape[1] # Used for model creation
        T = train_dataset[0][0].shape[2] # Used for model creation
        
        # Create the model 
        model = cf.get_model(C, T, train_config['hidden_space_dimension'])
        print("Model created")
        
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
        metadata = dict(train_config = train_config, dataset_config = dataset_config)
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dictionary correction (i.e. add the parameter select by the sweep to the dictionaries used during training)

def correct_dataset_config(sweep_config, dataset_config):
    key_to_copy = ['normalize_trials', 'filter_band']
    key_to_copy = ['normalize_trials']
    
    key_to_copy = []
        
    for key in key_to_copy: dataset_config[key] = sweep_config[key] 

def correct_train_config(sweep_config, train_config):
    key_to_copy = [
        'alpha', 'beta', 'gamma',
        'hidden_space_dimension', 
        'batch_size', 'epochs', 'use_scheduler', 
        'L2_loss_type', 'use_shifted_VAE_loss',
        'lr_decay_rate'
    ]
    key_to_copy = [
        'alpha', 'beta', 'gamma',
        'lr_decay_rate', 'hidden_space_dimension'
    ]

    for key in key_to_copy: train_config[key] = sweep_config[key] 

def correct_config(sweep_config, other_config):
    for key in other_config:
        if key in sweep_config:
            other_config[key] = sweep_config[key]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other function

def get_uniform_distribution(min:float, max:float, quantize = False) -> dict:
    tmp_dict = dict(
        distribution = 'q_uniform' if quantize else 'uniform',
        min = min,
        max = max
    )

    return tmp_dict
