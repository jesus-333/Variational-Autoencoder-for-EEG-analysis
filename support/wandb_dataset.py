"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the model with the wandb framework
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import wandb
import numpy as np
from scipy.io import loadmat

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Download dataset from artifact

def download_D2A_from_artifact(config):
    if config['type_dataset'] == 'train': version = 'v0'
    elif config['type_dataset'] == 'test': version = 'v1'
    else: raise ValueError("The parameter type_dataset must be train or test")
    
    with wandb.init(project = 'Jesus-Dataset', name = 'download_D2A_{}'.format(config['type_dataset']), config = config) as run:
        config = wandb.config

        artifact = run.use_artifact('jesus_333/Jesus-Dataset/D2A:{}'.format(version), type='Dataset')
        dataset_dir = artifact.download()

        for i in range(config['n_subject']):
            # Create path
            data_path = os.path.join(dataset_dir, '{}_data.mat'.format(i + 1))
            label_path = os.path.join(dataset_dir, '{}_label.mat'.format(i + 1))
            
            # Load data and label
            data = loadmat(data_path)['data']
            event_matrix = loadmat(label_path)['event_matrix']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Add dataset to artifact

def generate_D2A_artifact():
    """
    Create a wandb artifact with the dataset 2A (D2A) of the BCI competition IV
    """

    with wandb.init(project = 'Jesus-Dataset', job_type = "train", name = 'load_D2a') as run:
        path = 'Dataset/D2A/v1/'

        dataset_artifact_train = wandb.Artifact("D2A", type = "Dataset", description = "Training dataset 2A (D2A) of the BCI Competition IV")
        add_D2A_file_to_artifact(dataset_artifact_train, path, 'Train', 9)
        run.log_artifact(dataset_artifact_train)

        dataset_artifact_test = wandb.Artifact("D2A", type = "Dataset", description = "Testing dataset 2A (D2A) of the BCI Competition IV")
        add_D2A_file_to_artifact(dataset_artifact_test, path, 'Test', 9)
        run.log_artifact(dataset_artifact_test)
        
def add_D2A_file_to_artifact(artifact, path, dataset_type, n_subject):
    path = path + dataset_type + '/'
    
    # Iterate thorugh subject
    for i in range(n_subject):
        # String with the names of the file
        data_file_name = "{}_data.mat".format(i + 1)
        label_file_name = "{}_label.mat".format(i + 1)
        
        # Add data to artifact
        artifact.add_file(path + data_file_name)
        wandb.save(path + data_file_name)
        
        # Add label to artifact
        artifact.add_file(path + label_file_name)
        wandb.save(path + label_file_name)
