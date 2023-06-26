"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function used to check the dictionary of parameters
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Check config dataset 

def check_config_dataset(config):
    # Check the frequency filter settings
    if 'filter_data' not in config: 
        config['filter_data'] = False
        print('filter_data not specified. Set to False')

    if config['filter_data'] == True:
        if 'fmin' not in config or 'fmax' not in config: raise ValueError('If you want to filter the data you must specify the lower (fmin) and upper (fmax) frequency bands  of the filter')
    
    # Check the resampling settings
    if 'resample_data' not in config: 
        config['resample_data'] = False
        print("resample_data not specified. Set to False")

    if config['resample_data']:
        if 'resample_freq' not in config: raise ValueError('You must specify the resampling frequency (resample_freq)')
        if config['resample_freq'] <= 0: raise ValueError('The resample_freq must be a positive value')

    if config['use_moabb_segmentation']: 
        config['use_moabb_segmentation'] = False
        print('use_moabb_segmentation not specified. Set to False')

    if config['normalization_type'] not in config:
        config['normalization_type'] = 0
        print('normalization_type not specified. Set to 0 (no normalization)')

    if 'use_stft_representation' in config:
        if 'stft_parameters' not in config:
            raise ValueError("To transform the input through stft you must add dictionary with the parameters for the stft inside the dataset config")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Check config train 

def check_train_config(train_config : dict, model_artifact = None):
    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
    
    # Check if wandb is used during training
    if 'wandb_training' not in train_config: train_config['wandb_training'] = False
    if train_config['wandb_training'] == True and model_artifact is None: raise ValueError("If you want to train the model and load the data on wandb you must also pass an artifact to save the network") 
    
    # Path where save the network during training
    if 'path_to_save_model' not in train_config:
        print("path_to_save_model not found. Used current directory")
        train_config['path_to_save_model'] = "."
    
    if 'measure_metrics_during_training' not in train_config:
        print("measure_metrics_during_training not found. Set to False")
        train_config['measure_metrics_during_training'] = False

    if 'print_var' not in train_config:
        print("print_var not found. Set to True")
        train_config['print_var'] = True
