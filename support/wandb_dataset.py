"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the model with the wandb framework
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import sys
import os
import wandb
import numpy as np
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Add dataset to artifact# Add dataset to artifact

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
