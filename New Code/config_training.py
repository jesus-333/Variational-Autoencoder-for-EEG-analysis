"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config for the training functions
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_MBEEGNet_training():
    config = dict(
        # Training settings
        batch_size = 30,                    
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 10,                        # Number of epochs to train the model
        use_scheduler = False,              # Use the lr scheduler
        lr_decay_rate = 0.995,              # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer

        # Support stuff (device, log frequency etc)
        device = "cuda" if torch.cuda.is_available() else "cpu",
        epoch_to_save_model = 5,
        path_to_save_model = 'TMP_Folder',
        measure_metrics_during_training = True,
        repetition = 1,                     # Number of time to repeat the training 
        print_var = True,

        # (OPTIONAL) wandb settings
        wandb_training = False,             # If True track the model during the training with wandb
        project_name = "MBEEGNet",
        model_artifact_name = "MBEEGNet_test",    # Name of the artifact used to save the models
        log_freq = 1,
        notes = "",
        debug = True,                       # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )

    return config
