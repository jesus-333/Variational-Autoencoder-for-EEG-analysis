"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

File with the dictionary containing the config for the training
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#&& Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Function 

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
        batch_size = 32,                    
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 10,                        # Number of epochs to train the model
        use_scheduler = True,               # Use the lr scheduler
        gamma = 0.9,                        # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer
        # Support stuff (device, log frequency etc)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        log_freq = 1,
        epoch_to_save_model = 1,
        measure_accuracy_during_training = True,
        print_var = True,
        debug = True,                       # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )
    

    return train_config
