"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with support function relative to the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import wandb
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function to download/load file inside artifact

def add_file_to_artifact(artifact, file_name):
    artifact.add_file(file_name)
    wandb.save(file_name)

def download_artifacts_Ofner2017(version_list : list) :
    # Create a run to download the artifacts
    with wandb.init() as run :
        # Iterate over the list of versions
        for i in range(len(version_list)) :
            # Get the version
            version = int(version_list[i])

            # Create the artifact
            artifact = run.use_artifact('jesus_333/hvEEGNet_Ofner2017/hvEEGNet_Ofner2017_trained:v{}'.format(version), type='model')

            # Download the artifact
            artifact_dir = artifact.download()

            # Get metadata
            metadata = artifact.metadata

            # Get name training run
            name_training_run = metadata['training_config']['name_training_run']

            # For some training runs I have to manually rename in wandb
            # So the name_training_run in metadata is wrong
            if version == 11 : name_training_run = "S1_Colab_run_train_2"
            if version == 18 : name_training_run = "S9_TEST_DATA_run_train_3"
            if version == 19 : name_training_run = "S6_Colab_run_train_2"

            # Change folder name
            os.rename(artifact_dir, os.path.join(os.path.dirname(artifact_dir), name_training_run))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Function related to load classification metrics

def update_log_dict_metrics(metrics_dict, log_dict, label):
    # return accuracy, cohen_kappa, sensitivity, specificity, f1
    accuracy = metrics_dict['accuracy']
    cohen_kappa = metrics_dict['cohen_kappa']
    sensitivity = metrics_dict['sensitivity']
    specificity = metrics_dict['specificity']
    f1 = metrics_dict['f1']

    log_dict['accuracy_{}'.format(label)] = accuracy
    log_dict['cohen_kappa_{}'.format(label)] = cohen_kappa
    log_dict['sensitivity_{}'.format(label)] = sensitivity
