"""
Functions used for the servers in federated training.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import os
import torch
import numpy as np
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

from .. import wandb_support

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

        # Wandb artifact to save model weights
        self.model_artifact = wandb.Artifact('hvEEGNet_federated', type = "model",
                                        description = "hvEEGNet trained with federated training model",
                                        metadata = server_config)

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
            save_path = self.save_model_weights()

            # Add weight to wandb
            self.model_artifact.add_file(save_path)
            wandb.save(save_path)
            
            # TODO Remove in future. Used only to log something for the server at the end of each round
            avg_recon_loss = 0
            avg_kl_loss = 0
            avg_total_loss = 0
            
            # Iterate over clients
            for i in range(len(results)) :
                # Get metrics log dict for current client
                log_dict = results[i][1].metrics
                
                # Get id and number of training epoch
                client_id = log_dict['client_id']

                # Create epoch arrays
                training_epochs = np.arange(log_dict['epochs']) + 1

                # Extract losses
                recon_loss = self.extract_metric_from_log_dict(log_dict, 'train_loss_recon')
                kl_loss = self.extract_metric_from_log_dict(log_dict, 'train_loss_kl')
                total_loss = self.extract_metric_from_log_dict(log_dict, 'train_loss_total')

                # TODO remove
                avg_recon_loss += np.mean(recon_loss)
                avg_kl_loss += np.mean(kl_loss)
                avg_total_loss += np.mean(total_loss)

                # Plot(s) creation and log
                if self.server_config['log_loss_type'] == 1 : # Separate plots
                    self.create_and_log_plot([recon_loss], training_epochs, ['recon_loss'], client_id)
                    self.create_and_log_plot([kl_loss],    training_epochs, ['kl_loss'],    client_id)
                    self.create_and_log_plot([total_loss], training_epochs, ['total_loss'], client_id)
                elif self.server_config['log_loss_type'] == 2 : # Single plot 
                    self.create_and_log_plot([recon_loss, kl_loss, total_loss], training_epochs, ['recon_loss', 'kl_loss', 'total_loss'], client_id)
                elif self.server_config['log_loss_type'] == 3 : # Both previous option
                    self.create_and_log_plot([recon_loss], training_epochs, ['recon_loss'], client_id)
                    self.create_and_log_plot([kl_loss],    training_epochs, ['kl_loss'],    client_id)
                    self.create_and_log_plot([total_loss], training_epochs, ['total_loss'], client_id)
                    self.create_and_log_plot([recon_loss, kl_loss, total_loss], training_epochs, ['recon_loss', 'kl_loss', 'total_loss'], client_id)
            
            # TODO add a central dataset and computation at the end of each round
            # Log loss for the server
            avg_recon_loss /= len(results)
            avg_kl_loss /= len(results)
            avg_total_loss /= len(results)
            self.wandb_run.log({
                "server_recon_loss" : avg_recon_loss,
                "server_kl_loss" : avg_kl_loss,
                "server_total_loss" : avg_total_loss
            })
        
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
        save_path = f"{self.server_config['path_to_save_model']}/model_round_{self.count_rounds}.pth"
        torch.save(self.model.state_dict(), save_path)

        return save_path 

    def extract_metric_from_log_dict(self, log_dict : dict, metric_name : str) :
        """
        Since at the time of writing flower not allow to save entire array/list inside the log dict I have to save each epoch separately, i.e. with a different enty in the dictionary.
        With this method I will merge all the entry for a specific key in a list. 
        The list is then returned as numpy array
        """
        
        training_epochs = np.arange(log_dict['epochs']) + 1
        metric_list = []
        
        # Iterate over training epoch
        for i in range(len(training_epochs)) :
            # Get epoch and metric for the epoch
            current_epoch = training_epochs[i]
            metric_for_current_epoch = log_dict[f'{metric_name}_{current_epoch}']

            # Save metric
            metric_list.append(metric_for_current_epoch)
        
        return np.asarray(metric_list)

    def create_and_log_plot(self, metric_to_plot_list, training_epochs, metrics_name_list, client_id) :
        fig, _ = self.create_metric_plot(metric_to_plot_list, training_epochs, metrics_name_list, client_id)
        
        # Name for the plot to log
        if len(metrics_name_list) == 1 :
            metric_name = metrics_name_list[0]
        else : 
            metric_name = 'all'
        
        # Log the plot in wandb
        self.wandb_run.log({
            f"{client_id}/{metric_name}_round_{self.count_rounds}" : fig
        }, commit = False)

    def create_metric_plot(self, metric_to_plot_list : list, training_epochs, metrics_name_list : list, client_id : str) :
        """
        Create a plot with a loss from a specifi client
        """
        fig, ax = plt.subplots(1, 1, figsize = (16, 10))
        fontsize = 16

        for i in range(len(metrics_name_list)) :
            # Plot the metric
            ax.plot(training_epochs, metric_to_plot_list[i], label = metrics_name_list[i])

            ax.legend(fontsize = fontsize)
            ax.set_xlabel('Epoch', fontsize = fontsize)
            ax.set_ylabel('Loss', fontsize = fontsize)
            ax.grid(True)

        fig.tight_layout()

        return fig, ax
