"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Functions used to check the dictionary of parameters
"""

import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check config dataset

def check_config_dataset(config) -> None :
    # Check the frequency filter settings
    if 'filter_data' not in config:
        config['filter_data'] = False
        print('filter_data not specified. Set to False')

    if config['filter_data'] is True:
        if 'fmin' not in config or 'fmax' not in config: raise ValueError('If you want to filter the data you must specify the lower (fmin) and upper (fmax) frequency bands  of the filter')
    
    # Check the resampling settings
    if 'resample_data' not in config:
        config['resample_data'] = False
        print("resample_data not specified. Set to False")

    if config['resample_data']:
        if 'resample_freq' not in config: raise ValueError('You must specify the resampling frequency (resample_freq)')
        if config['resample_freq'] <= 0: raise ValueError('The resample_freq must be a positive value')

    if 'use_moabb_segmentation' not in config : 
        print('use_moabb_segmentation not specified in dataset config. Set to True')
        config['use_moabb_segmentation'] = True

    if 'n_samples_to_use' not in config : config['n_samples_to_use'] = -1

    # if config['use_moabb_segmentation']:
    #     config['use_moabb_segmentation'] = False
    #     print('use_moabb_segmentation not specified. Set to False. Ignore if you are not using moabb')

    # if 'normalization_type not in config:
    #     config['normalization_type'] = 0
    #     print('normalization_type not specified. Set to 0 (no normalization)')

    if config['train_trials_to_keep'] == [] : config['train_trials_to_keep'] = None

    if 'use_stft_representation' in config:
        if config['use_stft_representation'] and 'stft_parameters' not in config:
            raise ValueError("To transform the input through stft you must add dictionary with the parameters for the stft inside the dataset config")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Check config train

def check_train_config(train_config : dict, model_artifact = None) -> None :
    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
       
    # Check if wandb is used during training
    if 'wandb_training' not in train_config : train_config['wandb_training'] = False
    # if train_config['wandb_training'] is True and model_artifact is None :
    #     raise ValueError("If you want to train the model and load the data on wandb you must also pass an artifact to save the network")
    
    # Path where save the network during training
    if 'path_to_save_model' not in train_config:
        print("path_to_save_model not found. Used current directory")
        train_config['path_to_save_model'] = "."
    else :
        # Create the folder to save the model weights
        os.makedirs(train_config['path_to_save_model'], exist_ok = True)
    
    if 'measure_metrics_during_training' not in train_config:
        print("measure_metrics_during_training not found. Set to False")
        train_config['measure_metrics_during_training'] = False

    if 'print_var' not in train_config:
        print("print_var not found. Set to True")
        train_config['print_var'] = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Check model config

def check_model_config_EEGNet(model_config : dict) -> None : 
    # In EEGNet class to skip the pooling layer the values of p_kernel_1/p_kernel_2 must be set to None
    # But toml format not support None/nill values. So for the pool kernel I used the value -1 inside the toml file
    # Here, if I find a -1 in p_kernel_1/p_kernel_2, I change the value to None
    
    if type(model_config['p_kernel_1']) is int : 
        if model_config['p_kernel_1'] <= -1 : 
            model_config['p_kernel_1'] = None
            print("Find invalid value for p_kernel_1. Set to None")
    if type(model_config['p_kernel_2']) is int : 
        if model_config['p_kernel_2'] <= -1 : 
            model_config['p_kernel_2'] = None
            print("Find invalid value for p_kernel_2. Set to None")

    # Toml load list of mulitple element as python list
    # Some of this parametersa are used as input of Torch layer that require tuple and not list
    # So I convert each list in a tuple 
    key_list = ['c_kernel_1', 'c_kernel_2', 'c_kernel_3', 'p_kernel_1', 'p_kernel_2'] 
    for key in key_list :
        if model_config[key] is not None :
            model_config[key] = tuple(model_config[key])

def check_model_config_hvEEGNet(model_config : dict) -> None:
    check_model_config_EEGNet(model_config['encoder_config'])   

    if model_config['encoder_config']['flatten_output'] : 
        print("Automatically set flatten_output to False. For hvEEGNet flatten_output must be set to False.")
        model_config['encoder_config']['flatten_output'] = False

    if model_config['parameters_map_type'] < 0 or model_config['parameters_map_type'] > 3 :
        raise ValueError("Invalid value for parameters_map_type. Value must be between 0 and 3. Current value is {}".format(model_config['parameters_map_type']))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_server_config(server_config) -> None : 
    """
    Check the dictionary used for the server in the federated training
    """

    if 'notes' not in server_config['wandb_config'] : 
        server_config['wandb_config']['notest'] = 'No additional notes.'

    if 'name_training_run' not in server_config['wandb_config'] or len(server_config['wandb_config']['name_training_run']) == 0: 
        print('No name for the wandb run provided. Set to None')
        server_config['wandb_config']['name_training_run '] = None

    if 'path_to_save_model' not in server_config :
        raise ValueError("The key path_to_save_model is not specified in the config. You must specified where to save the model weights")

    if 'model_config' not in server_config :
        raise ValueError('The key model_config is not specified in the config. The server need a copy of the model config from the client to create a local copy of the model')

    check_model_config_hvEEGNet(server_config['model_config'])

