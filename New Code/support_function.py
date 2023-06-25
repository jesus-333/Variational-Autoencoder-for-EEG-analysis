"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Minor support function used in the various script
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_activation(activation_name: dict):
    """
    Receive a string and return the relative activation function in pytorch.
    Implemented for relu, elu, selu, gelu
    """

    if activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'selu':
        return nn.SELU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        error_message = 'The activation must have one of the following string: relu, elu, selu, gelu'
        raise ValueError(error_message)

def get_dropout(prob_dropout: float, use_droput_2d : bool):
    if use_droput_2d:
        return nn.Dropout2d(prob_dropout)
    else: 
        return nn.Dropout(prob_dropout)


def count_trainable_parameters(layer):
    n_paramters = sum(p.numel() for p in  layer.parameters() if p.requires_grad)
    return n_paramters


def split_dataset(full_dataset, percentage_split):
    """
    Split a dataset in 2 for train and validation
    """

    size_train = int(len(full_dataset) * percentage_split) 
    size_val = len(full_dataset) - size_train
    train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [size_train, size_val])
    
    return train_dataset, validation_dataset        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Check config dictionary

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
        
    # if 'subject_by_subject_normalization' not in config: config['subject_by_subject_normalization'] = False

    if config['use_moabb_segmentation']: 
        config['use_moabb_segmentation'] = False
        print('use_moabb_segmentation not specified. Set to False')

    if config['normalization_type'] not in config:
        config['normalization_type'] = 0
        print('normalization_type not specified. Set to 0 (no normalization)')
    else:
        if config['normalization_type'] == 1 and 'stft_parameters' not in config:
            raise ValueError("normalization_type = 1 require stft parameter")
