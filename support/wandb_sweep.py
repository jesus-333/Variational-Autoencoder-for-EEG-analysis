"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain function used to config the wandb sweep and during the sweep train
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import wandb
import torch
from torch.utils.data import DataLoader

import config_file as cf
from wandb_training import train_cycle, add_model_to_artifact

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train function

def train_sweep(config = None):
    # Get other config not included in the ones used for the sweep
    dataset_config = cf.get_dataset_config()
    train_config = cf.get_train_config()

    with wandb.init(project = "VAE_EEG", job_type = "train", config = config) as run:
        config = wandb.config

        # "Correct" dictionaries
        correct_dataset_config(config, dataset_config)
        correct_train_config(config, train_config)
        
        # Get the training data
        train_dataset, validation_dataset = cf.get_train_data(dataset_config)
        
        # Create dataloader
        train_dataloader        = DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        validation_dataloader   = DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
        loader_list             = [train_dataloader, validation_dataloader]
        
        # Get test data
        # test_loader_list = cf.get_subject_data(dataset_config, 'test', return_dataloader = True, batch_size = train_config['batch_size'])

        # Variables
        C = train_dataset[0][0].shape[1] # Used for model creation
        T = train_dataset[0][0].shape[2] # Used for model creation
     
        # Create the model 
        model = cf.get_model(C, T, train_config['hidden_space_dimension'])
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = train_config['lr'], 
                                      weight_decay = train_config['optimizer_weight_decay'])

        # Setup lr scheduler
        if train_config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
        else:
            lr_scheduler = None
        print("5")
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
    for key in key_to_copy: dataset_config[key] = sweep_config[key] 

def correct_train_config(sweep_config, train_config):
    key_to_copy = [
        'alpha', 'beta', 'gamma',
        'hidden_space_dimension', 
        'batch_size', 'epochs', 'use_scheduler', 'L2_loss_type',
        'lr_decay_rate'
    ]

    for key in key_to_copy: train_config[key] = sweep_config[key] 


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other function

def get_uniform_distribution(min:float, max:float) -> dict:
    tmp_dict = dict(
        distribution = 'uniform',
        min = min,
        max = max
    )

    return tmp_dict

