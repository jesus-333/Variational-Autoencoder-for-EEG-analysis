"""
Functions used for the servers in federated training.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict

try :
    import flwr
except :
    raise ImportError("To use the federated functions you need the flower framework. More info here https://pypi.org/project/flwr")

try :
    import wandb
except :
    raise ImportError("Failed import of wandb. The wandb_server class will not work.")

from ... import check_config
from ...model import hvEEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters conversion from/to numpy array

def get_weights(model : torch.nn.Module) -> list:
    """
    Given a PyTorch model extract the weights/parameters, convert them in a numpy array and save them to a list

    @param model : PyTorch model

    @return weights_list : (list) List where each element is a group of weights of a specific layer/module of the Pytorch model
    """

    weights_list = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return weights_list

def get_random_weights_hvEEGNet(model_config : dict) -> list:
    """
    Return the weights of a non traiend hvEEGNet created with the parameters specified in model_config.

    @param model_config : (dict) Dictionary with all the settings for hvEEGNet

    @return weights_list : (list) List where each element is a group of weights of a specific layer/module of hvEEGNet model
    """

    return get_weights(hvEEGNet.hvEEGNet_shallow(model_config))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Server function

class FedAvg_with_wandb(flwr.server.strategy.FedAvg):
    def __init__(self, server_config : dict()) :
        super().__init__()
        self.server_config = server_config 
        
        # Check config
        check_config.check_server_config(server_config)
        
        # Create model
        self.model = hvEEGNet.hvEEGNet_shallow(server_config['model_config'])
        
        # Create wandb run
        self.wandb_run = wandb.init(project = server_config['wandb_config']['project_name'], 
                                    job_type = "train", config = server_config, 
                                    notes = server_config['wandb_config']['notes'], 
                                    name = server_config['wandb_config']['name_training_run']
                                    )

        self.count_rounds = 0
        self.tot_rounds = server_config['num_rounds']

    def aggregate_fit(self, server_round: int, results, failures) :
        """
        aggregate the results from the results from the clients and upload the results in wandb
        """

        self.count_rounds += 1

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:

            # Convert `Parameters` to `List[np.ndarray]`
            model_weights = flwr.common.parameters_to_ndarrays(aggregated_parameters) 
            self.model_weights  = model_weights

            # Save weights
            self.save_model_weights()

            for i in range(len(results)) :
                metrics_from_training = results[i][1].metrics
                print(metrics_from_training)
        
            if self.count_rounds == self.tot_rounds :
                print("End training rounds")
                self.wandb_run.finish()
        else :
            model_weights = None

        return aggregated_parameters, aggregated_metrics

    def save_model_weights(self) :
        """
        Save the model after a training round in the path specified in server_config['path_to_save_model']
        """
        # Create folder specified in the path
        os.makedirs(self.server_config['path_to_save_model'], exist_ok = True)

        # Convert `List[np.ndarray]` to PyTorch`state_dict`
        params_dict = zip(self.model.state_dict().keys(), self.model_weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Save the model

    def create_plot_loss_client(self, loss_to_plot, label : str) :
        """
        Create a plot with a loss from a specifi client
        """
        fig, ax = plt.subplots(1, 1)
