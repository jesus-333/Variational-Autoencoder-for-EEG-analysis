"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train the various network and save the results with the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import wandb

# Custom functions
from . import train_generic

# Config files
from ..config import config_model as cm
from ..config import config_dataset as cd 
from ..config import config_training as ct

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_wandb(model_name, dataset_config, train_config, model_config):
    notes = train_config['notes']
    name = train_config['name'] if 'name' in train_config else None

    wandb_config = dict(
        dataset = dataset_config,
        train = train_config,
        model = model_config
    )

    with wandb.init(project = train_config['project_name'], job_type = "train", config = wandb_config, notes = notes, name = name) as run:
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)
        
        # Train the model
        model = train_generic.train_and_test_model(model_name, dataset_config, train_config, model_config, model_artifact)
        
        # Log the model artifact
        run.log_artifact(model_artifact)

        return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main_EEGNet_classifier(): 
    dataset_config = cd.get_moabb_dataset_config([3])
    
    train_config = ct.get_config_classifier()
    train_config['wandb_training'] = True

    C = 22
    T = 512 
    model_config = cm.get_config_EEGNet_stft_classifier(C, T, 22)
    
    model = train_wandb('EEGNet', dataset_config, train_config, model_config)
    
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_dict_for_vEEGNet(subj_list : list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)
    
    train_config = ct.get_config_vEEGNet_training()
    train_config['wandb_training'] = True
    train_config['model_artifact_name'] = 'vEEGNet'

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    hidden_space_dimension = 64
    type_encoder = 0
    type_decoder = 0
    model_config = cm.get_config_vEEGNet(C, T, hidden_space_dimension, type_encoder, type_decoder)
    
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    return dataset_config, train_config, model_config

def main_vEEGNet_default(subj_list: list):
    """
    Train vEEGNet with the value specified inside the config files
    """
    dataset_config, train_config, model_config = get_config_dict_for_vEEGNet(subj_list)
    return main_vEEGNet(dataset_config, train_config, model_config)

def main_vEEGNet(dataset_config : dict, train_config : dict, model_config):
    """
    Pass the config and train the model with wandb.
    (This method is also used in colab for training instead of main_vEEGNet_default because in colab it is not possible to modify other files)
    """

    model = train_wandb('vEEGNet', dataset_config, train_config, model_config)
    
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_dict_for_hvEEGNet_shallow(subj_list : list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)
    
    train_config = ct.get_config_vEEGNet_training()
    train_config['wandb_training'] = True
    train_config['model_artifact_name'] = 'hvEEGNet_shallow'

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    type_decoder = 0
    parameters_map_type = 0
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)
    
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    return dataset_config, train_config, model_config

def main_hvEEGNet_shallow(dataset_config, train_config, model_config):

    model = train_wandb('hvEEGNet_shallow', dataset_config, train_config, model_config)
    
    return model
