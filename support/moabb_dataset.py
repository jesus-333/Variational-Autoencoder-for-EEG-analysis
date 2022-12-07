#%% Imports

import numpy as np
import moabb.datasets as mb
import moabb.paradigms as mp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dataset 2A BCI Competition IV

def get_D2a(config):
    check_config(config)
    
    # Select the dataset
    dataset = mb.BNCI2014001()

    # Select the paradigm (i.e. the object to download the dataset)
    paradigm = mp.MotorImagery()

    # Get the raw data
    raw_data = get_raw_data(dataset, paradigm, config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other

def get_raw_data(dataset, paradigm, config):
    """
    Return the raw data from the moabb package of the spcified dataset and paradigm
    N.b. dataset and paradigm must be object of the moabb library
    """

    if config['resample']: paradigm.resample = config['resampling_freq']

    if config['filter_data']: 
        paradigm.fmin = config['fmin']
        paradigm.fmax = config['fmax']
    
    paradigm.n_classes = config['n_classes']

    # Get the raw data
    raw_data = paradigm.get_data(dataset = dataset, subjects = config['subjects_list'])

    return raw_data

def check_config(config):
    # Check the frequency filter settings
    if 'filter_data' not in config: config['filter_data'] = False

    if config['filter_data'] == True:
        if 'filter_freq_band' not in config: raise ValueError('If you want to filter the data you must specify the lower and upper frequency band of the filter')
    
    # Check the resampling settings
    if 'resampling_data' not in config: config['resampling_data'] = False
    if 'resampling_freq' not in config: config['resampling_data'] = False
    if config['resampling_freq'] <= 0: raise ValueError('The resampling_freq must be a positive value')
