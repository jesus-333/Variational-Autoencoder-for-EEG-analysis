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

from support_training import loadDatasetD2, computeTrialD2, saveTrialsForSubject, saveTrialsAllTogether
from support_function import filterSignal, resampling

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Download dataset from artifact

def download_D2A_from_artifact(config):
    if config['type_dataset'] == 'train': version = 'v2'
    elif config['type_dataset'] == 'test': version = 'v3'
    else: raise ValueError("The parameter type_dataset must be train or test")
    
    with wandb.init(project = 'Jesus-Dataset', name = 'download_D2A_{}'.format(config['type_dataset']), config = config) as run:
        config = wandb.config

        artifact = run.use_artifact('jesus_333/Jesus-Dataset/D2A:{}'.format(version), type='Dataset')
        dataset_dir = artifact.download()

        for i in range(config['n_subject']):
            load_and_convert_D2A_data(dataset_dir, config, i)

def load_and_convert_D2A_data(dataset_dir, config, idx):
    """
    Read the data from the dataset 2a from the file and save them in separate file for each trials and subject
    """

    # Create path
    data_path       = os.path.join(dataset_dir, '{}_data.mat'.format(idx + 1))
    label_path      = os.path.join(dataset_dir, '{}_label.mat'.format(idx + 1))
    true_label_path = os.path.join(dataset_dir, 'A0{}E.mat'.format(idx + 1))

    # Load data and label
    data = loadmat(data_path)['data']
    event_matrix = loadmat(label_path)['event_matrix']
    
    # Filtering
    if(config['filtering_data']):
        data = filterSignal(data, 250, config['fmin'], config['fmax'])
    
    # Compute trials
    trials, labels = computeTrialD2(data, event_matrix, 250, start_second = 2, end_second = 6)
    
    # Downsampling
    if(config['resample_data']): 
        trials =  resampling(trials, 250, config['resample_freq'], axis = 2)
    
    # Get true label (Only for test set)
    if config['dataset_type'] == 'Test':
        labels = np.squeeze(loadmat(true_label_path)['classlabel'])

    # Command fro v1 to v2
    saveTrialsForSubject(config['folder_to_save'], idx, trials, labels)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Add dataset to artifact (VERSION 1)

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

        # dataset_artifact_label = wandb.Artifact("D2A", type = "Dataset", description = "True labels of dataset 2A (D2A) of the BCI Competition IV")
        # add_true_label_file(dataset_artifact_label, 9)
        # run.log_artifact(dataset_artifact_label)
        
def add_D2A_file_to_artifact(artifact, path, dataset_type, n_subject):
    path = path + dataset_type + '/'
    
    # Iterate thorugh subject
    for i in range(n_subject):
        # String with the names of the file
        data_file_name = "{}_data.mat".format(i + 1)
        label_file_name = "{}_label.mat".format(i + 1)
        if dataset_type == 'Train':
            true_label_file_name = "A0{}T.mat".format(i + 1)
        else:
            true_label_file_name = "A0{}E.mat".format(i + 1)

        # Add data to artifact
        artifact.add_file(path + data_file_name)
        wandb.save(path + data_file_name)
        
        # Add label to artifact
        artifact.add_file(path + label_file_name)
        wandb.save(path + label_file_name)

        # Add true label to artifact
        artifact.add_file(path + true_label_file_name)
        wandb.save(path + true_label_file_name)

def add_true_label_file(artifact, n_subject):
    path = 'C:/Users/albi2/Documents/GitHub/Variational-Autoencoder-for-EEG-analysis/Dataset/D2A/true_labels_2a'

    for i in range(n_subject):
        # Train label
        file_name = 'A0{}T.mat'
        artifact.add_file(path + file_name)
        wandb.save(path + file_name)
        
        # Test label
        file_name = 'A0{}E.mat'
        artifact.add_file(path + file_name)
        wandb.save(path + file_name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Add dataset to artifact (VERSION 2)
# TODO finis

def generate_D2A_artifact_V2(path):
    with wandb.init(project = 'Jesus-Dataset', job_type = "Load_dataset", name = 'load_D2a') as run:
        dataset_artifact_train = wandb.Artifact("D2A_v2", type = "Dataset", description = "Training dataset 2A (D2A) of the BCI Competition IV (VERSION 2)")
        add_D2A_file_to_artifact(dataset_artifact_train, path, 'Train', 9)
        run.log_artifact(dataset_artifact_train)

def add_D2A_file_to_artifact_V2(dataset_type):
    path = path + dataset_type + '/'
