"""
Script to be used for the server during federated training.
In this case we used the ServerApp

N.b. This script is unfinished and currently not works.

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

def server_fn(context: flwr.common.Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = federated_training.get_random_weights_hvEEGNet()
    parameters = flwr.common.ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = flwr.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = flwr.server.ServerConfig(num_rounds=num_rounds)

    return flwr.server.ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = flwr.server.ServerApp(server_fn=server_fn)

