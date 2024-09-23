"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train the various network and save the results with the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import wandb
import torch

# Custom functions
from . import train_generic
from .. import check_config

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

def train_wandb_V1(model_name : str, dataset_config : dict, train_config : dict, model_config : dict):
    """
    Train a model with the data downloaded through the moabb tools.
    """

    notes = train_config['notes']
    name = train_config['name_training_run'] if 'name' in train_config else None

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


def train_wandb_V2(model_name : str, train_config : dict, model_config : dict, train_dataset, validation_dataset, dataset_config : dict = None):
    """
    Train a model with the data provided by the user and log everything on wandb.

    @param model_name: string with the name of the model.
    @param train_config: dictionary with all the hyperparameter for the training. Check the README_training for more info.
    @param model_config: dictionary with all the parameter to use in the creation of the model. Check README_model for more info.
    @param train_dataset : dataset with the training data. Must by implemented with the class EEG_dataset inside library.dataset_time
    @param validation_dataset : dataset with the validation data. Must by implemented with the class EEG_dataset inside library.dataset_time
    @param dataset_config : (dict) Dictionary with all the parameters used for the data preprocess and dataset creation. Not used during the training but if passed it is logged in wandb. Default to None

    @return: the trained model.
    """
    
    if 'hvEEGNet' in model_name : check_config.check_model_config_hvEEGNet(model_config)
    check_config.check_train_config(train_config)

    notes = train_config['notes'] if 'notes' in train_config else 'No notes in train_config'
    name = train_config['name_training_run'] if 'name_training_run' in train_config else None
    
    # Add train and model config to wandb dict so they are saved and logged in wandb
    wandb_config = dict(
        train = train_config,
        model = model_config
    )
    
    # If not None add also the dataset_config 
    if dataset_config is not None : wandb_config['dataset'] = dataset_config
    
    # Create dataloader
    print(len(train_dataset), train_config['batch_size'])
    print(len(validation_dataset), train_config['batch_size'])
    train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
    validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
    loader_list             = [train_dataloader, validation_dataloader]

    # Create model
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model = train_generic.get_untrained_model(model_name, model_config)
    model.to(train_config['device'])

    # Declare loss function
    loss_function = train_generic.get_loss_function(model_name, train_config)

    # Get loss function
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = train_config['lr'],
                                  weight_decay = train_config['optimizer_weight_decay']
                                  )

    # Setup lr scheduler
    if train_config['use_scheduler'] :
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
    else:
        lr_scheduler = None

    with wandb.init(project = train_config['project_name'], job_type = "train", config = wandb_config, notes = notes, name = name) as run:
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)
        
        # Train the model
        model = train_generic.train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler, model_artifact)
        
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
    
    model = train_wandb_V1('EEGNet', dataset_config, train_config, model_config)
    
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
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf)
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

    model = train_wandb_V1('vEEGNet', dataset_config, train_config, model_config)
    
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
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf)
    type_decoder = 0
    parameters_map_type = 0
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)
    
    train_config['measure_metrics_during_training'] = model_config['use_classifier']
    train_config['use_classifier'] = model_config['use_classifier']

    return dataset_config, train_config, model_config

def main_hvEEGNet_shallow(dataset_config, train_config, model_config):

    model = train_wandb_V1('hvEEGNet_shallow', dataset_config, train_config, model_config)
    
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_dict_for_classifier_v1(subj : int, path_weight_hvEEGNet : str):
    dataset_config, _, hvEEGNet_config = get_config_dict_for_hvEEGNet_shallow([subj])
    classifier_config = cm.get_config_classifier_v1()

    train_config = ct.get_config_classifier()
    train_config['wandb_training'] = True
    train_config['model_artifact_name'] = 'classifier'



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if __name__ == '__main__':
    pass
    # model = main_EEGNet_classifier()
