"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config for the training functions
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def get_config_classifier():
    config = dict(
        # Training settings
        batch_size = 30,                    
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 300,                        # Number of epochs to train the model
        use_scheduler = False,              # Use the lr scheduler
        lr_decay_rate = 0.995,              # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer

        # Support stuff (device, log frequency etc)
        device = "cuda" if torch.cuda.is_available() else "cpu",
        # device = "cpu",
        epoch_to_save_model = 5,
        path_to_save_model = 'TMP_Folder',
        measure_metrics_during_training = True,
        repetition = 1,                     # Number of time to repeat the training 
        print_var = True,

        # (OPTIONAL) wandb settings
        wandb_training = False,             # If True track the model during the training with wandb
        project_name = "ICT4AWE_Extension",
        model_artifact_name = "EEGNet_stft",    # Name of the artifact used to save the models
        log_freq = 1,
        notes = "",
        debug = True,                       # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )

    return config

def get_config_vEEGNet_training():
    config = dict(
        # Training settings
        batch_size = 30,                    
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 10,                        # Number of epochs to train the model
        use_scheduler = True,              # Use the lr scheduler
        lr_decay_rate = 0.995,              # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer
        alpha = 1,
        beta = 1,
        gamma = 1,

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

#%% Sweep 

def get_config_sweep(metric_name, metric_goal):
    if metric_goal != 'maximize' and metric_goal != 'minimize':
        raise ValueError('The metric_goal parameter must have value \'maximize\' or \'minimize\'')

    sweep_config = dict(
        # Fields needed by wandb
        method = 'random',
        metric = dict(
            name = metric_name,
            goal = metric_goal
        ),
        parameters = dict(
            epochs = dict(
                values = [200, 300, 500],
                # values = [1,2,3] # Used for debug
            ),
            lr_decay_rate = dict(
                values = [1, 0.995],
            ),
            batch_size = dict(
                values = [30, 40, 50],
            ),
            subjects_list = dict(
                values = [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
            ),
            channels_list = dict(
                values = [['C3', 'Cz', 'C4'], ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6','CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']],
            ),
            window = dict(
                values = ['hann', ('gaussian', 1), ('gaussian', 2)],
            ),
            kernel = dict(
                values = [(3,3), (5,5), (7,7), (9,9)],
            ),
            filter_1 = dict(
                values = [8, 16, 32],
            ),
            D = dict(
                values = [2, 4, 8, 16],
            ),
            prob_dropout = dict(
                values = [0.3, 0.4, 0.5],
            ),
            use_dropout_2d = dict(
                values = [True, False],
            )
        ),
    )

    return sweep_config