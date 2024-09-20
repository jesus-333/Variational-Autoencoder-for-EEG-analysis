"""
Script used to train hvEEGNet with federated learning if multiple device are not available

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import toml
import flwr

from library.analysis import support
from library.training.federated import server, simulation

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_list = [4, 5, 6, 7, 8, 9]
subj_list = [2, 5, 8]

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
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # TODO Change to random sampling. Reduce dataset size
    # train_dataset.data = train_dataset.data[0:int(259/3)]
    
    # Save dataset in list
    train_dataset_list.append(train_dataset)
    validation_dataset_list.append(validation_dataset)
    
# Update model_config. Notes that this work because all model config point to the same dictionary in the memory
server_config['model_config']['encoder_config']['C'] = train_dataset.data.shape[2]
server_config['model_config']['encoder_config']['T'] = train_dataset.data.shape[3]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Simulation

# Generate client function for each subject
client_function = simulation.generate_client_function_hvEEGNet_training(model_config_list, train_config_list, 
                                                                        train_dataset_list, validation_dataset_list
                                                                        )
# Run simulation
flwr.simulation.start_simulation(
    client_fn = client_function,
    num_clients = len(subj_list),
    config = flwr.server.ServerConfig(
        num_rounds = server_config['num_rounds']
    ),  
    strategy = strategy,
    client_resources = {
        "num_cpus": 2,
        "num_gpus": 1,
    }, 
)
