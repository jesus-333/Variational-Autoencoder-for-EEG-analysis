"""
Script used to train hvEEGNet with federated learning if multiple device are not available

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml

from library.analysis import support
from library.training.federated import server, simulation

from library.config import config_dataset as cd
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_list = [4, 5, 6, 7, 8, 9]

path_server_config_file = 'training_scripts/config/federated_server.toml'
path_client_config_file = 'training_scripts/config/federated_client.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get data

train_dataset_list = []
validation_dataset_list = []

for i in range(subj_list) :
    subj = subj_list[i]

    # Get dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    # Save dataset in list
    train_dataset_list.append(train_dataset_list)
    validation_dataset.append(validation_dataset)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

config = toml.load(path_server_config_file)
strategy = server.FedAvg_with_wandb(config)
