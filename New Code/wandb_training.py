"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train the various network and save the results with the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch
import sys
import wandb

# Custom functions
import train_MBEEGNet

# Config files
import config_model
import config_dataset
import config_training

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_wandb_MBEEGNet(dataset_config, train_config, model_config):
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

        train_MBEEGNet.train_and_test_model(dataset_config, train_config, model_config, model_artifact)
