"""
Script used to train hvEEGNet with federated learning if multiple device are not available

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml
import flwr

from library.dataset import preprocess as pp
from library.training.federated import server, simulation

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_list = [4, 5, 6, 7, 8, 9]
subj_list = [2, 8]

path_server_config_file = 'training_scripts/config/federated/server.toml'
path_client_config_file = 'training_scripts/config/federated/client.toml'
path_train_config = 'training_scripts/config/federated/training.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Server setup

# Get server config
server_config = toml.load(path_server_config_file)
server_config['subj_list'] = subj_list

# Create server
strategy = server.FedAvg_with_wandb(server_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get data

model_config_list = []
train_config_list = []

train_dataset_list = []
validation_dataset_list = []

for i in range(len(subj_list)) :
    subj = subj_list[i]

    # Save model config
    model_config_list.append(server_config['model_config'])
    # (Note that in this way the dictionary is shared, i.e. you copy only the pointer and each modification at server_config['model_config'] will affect every element of model_config_list)
    # This is still not a problem since all the clients share the same config

    # Save train config
    train_config = toml.load(path_train_config)
    train_config['clint_id'] = 'S{}'.format(subj)
    train_config_list.append(train_config)

    # Get dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Save dataset in list
    train_dataset_list.append(train_dataset)
    validation_dataset_list.append(validation_dataset)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

# Generate client function for each subject
# client_function = simulation.generate_client_function_hvEEGNet_training(model_config_list, train_config_list, 
#                                                                         train_dataset_list, validation_dataset_list
#                                                                         )
# # Run simulation
# flwr.simulation.start_simulation(
#     client_fn = client_function,
#     num_clients = len(subj_list),
#     config = flwr.server.ServerConfig(
#         num_rounds = server_config['num_rounds']
#     ),  
#     strategy = strategy,
#     client_resources = {
#         "num_cpus": 2,
#         "num_gpus": 1,
#     }, 
# )
