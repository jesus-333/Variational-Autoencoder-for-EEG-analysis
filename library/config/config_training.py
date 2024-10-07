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
    """
    Used to train EEGNet and MBEEGNet model
    """
    config = dict(
        # Training settings
        batch_size = 20,
        lr = 1e-3,                          # Learning rate (lr)
        epochs = 300,                        # Number of epochs to train the model
        use_scheduler = False,              # Use the lr scheduler
        lr_decay_rate = 0.999,              # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer

        # Support stuff (device, log frequency etc)
        device = "cuda" if torch.cuda.is_available() else "cpu",
        # device = "cpu",
        epoch_to_save_model = 5,
        path_to_save_model = 'TMP_Folder',
        measure_metrics_during_training = True,
        repetition = 1,                     # Number of time to repeat the training 
        use_classifier = True,              # Do NOT CHANGE and keep True. Needed for metric computation in the train function (train_generic.py) 
        print_var = True,

        # (OPTIONAL) wandb settings
        wandb_training = False,             # If True track the model during the training with wandb
        project_name = "hvEEGNet_extension",
        model_artifact_name = "artifact_name",    # Name of the artifact used to save the models
        log_freq = 1,                           # How often log gradients and parameters of the tracked model. Ignore and left to 1.
        name_training_run = None,               # Name of the training run. If None wandb will assign a random name.
        notes = "",                             # If you want to add specific note for a specific training run modify this field.
        debug = True,                           # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )

    return config

def get_config_vEEGNet_training() -> dict:
    config = dict(
        # Training settings
        batch_size = 30,
        lr = 1e-2,                          # Learning rate (lr)
        epochs = 3,                         # Number of epochs to train the model
        use_scheduler = True,               # Use the lr scheduler
        lr_decay_rate = 0.999,              # Parameter of the lr exponential scheduler
        optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer
        alpha = 1,                          # Multiplier of the reconstruction error
        beta = 1,                           # Multiplier of the KL
        gamma = 1,                          # Multiplier of the classification error (if you also use a classifier). It's completely independent from the gamma_dtw, they simply share a similar name.
        recon_loss_type = 1,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
        edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
        average_channels = False,
        gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min

        # Support stuff (device, log frequency etc)
        device = "cuda" if torch.cuda.is_available() else "cpu",
        epoch_to_save_model = 5,                    # How often save the model weights (e.g. 5 means that the weights are saved every 5 epochs)
        path_to_save_model = 'TMP_Folder',          # Folder where the weights of the model will be saved during training
        use_classifier = False,                     # Ignore. It is used only if the model has a classifier. In this way the code know that during the training it also need to compute the classification error
        measure_metrics_during_training = True,     # Ignore. If True measuere accuracy and other metrics during the training. Works only if the model has a classifier
        print_var = True,

        # (OPTIONAL) wandb settings
        wandb_training = False,             # If True track the model during the training with wandb
        project_name = "TMP_Project",       # Name of wandb project
        model_artifact_name = "TMP_NAME",   # Name of the artifact used to save the model
        log_freq = 1,                       # How often log gradients and parameters of the tracked model. Ignore and left to 1.
        name_training_run = None,           # Name of the training run. If None wandb will assign a random name.
        notes = "",                         # If you want to add specific note for a specific training run modify this field.
        debug = True,                       # Set True if you are debuggin the code (Used to delete debug run from wandb)
    )

    return config

def get_config_hierarchical_vEEGNet_training() -> dict:
    """
    Return the config require for the hvEEGNet training.
    Notes that the the configs required are the same of the standard vEEGNet.
    """
    return get_config_vEEGNet_training()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Sweep 

def get_config_sweep_EEGNet_classifier(metric_name, metric_goal):
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

