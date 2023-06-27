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
import train_generic

# Config files
import config_model as cm
import config_dataset as cd 
import config_training as ct

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_wandb_EEGNet(dataset_config, train_config, model_config):
    notes = train_config['notes']

    wandb_config = dict(
        dataset = dataset_config,
        train = dataset_config,
        model = model_config
    )

    with wandb.init(project = train_config['project_name'], job_type = "train", config = wandb_config, notes = notes) as run:
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)

        model = train_generic.train_and_test_model('EEGNet', dataset_config, train_config, model_config, model_artifact)
        
        return model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main_EEGNet_classifier():
    dataset_config = cd.get_moabb_dataset_config([2])
    dataset_config['stft_parameters'] = cd.get_config_stft()
    
    train_config = ct.get_config_classifier()
    train_config['wandb_training'] = True
    
    C = 22
    T = 512 
    model_config = cm.get_config_EEGNet_stft_classifier(C, T, 22)
    
    model = train_wandb_EEGNet(dataset_config, train_config, model_config)
    
    return model

if __name__ == '__main__':
    model = main_EEGNet_classifier()
