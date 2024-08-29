"""
Script to be used for the server during federated training

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports


from library.training import federated_training

import sys
import flwr 

try :
    import toml
except :
    raise ImportError("The training config are saved in a toml file. To read it you need the toml library. See here for more info https://pypi.org/project/toml/")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get config

path_config_file = ""

if path_config_file == "" :
    print("Path for the config file not specified in the python script. Check if passed as argument")
    if len(sys.argv) == 1 : 
        print("Path for the config file not specified as argument. Used the default path \"training_scripts/config/federated_server.toml\"")
        path_config_file = "training_scripts/config/federated_server.toml" 
    else :
        print("Path passed as argument.")
        path_config_file = sys.argv[1]

config = toml.load(path_config_file)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Start server 
flwr.server.start_server(
    server_address = "0.0.0.0:{}".format(config['server_port']),
    config = flwr.server.ServerConfig(num_rounds = config['num_rounds']),
    strategy = federated_training.FedAvg_with_wandb()
)
