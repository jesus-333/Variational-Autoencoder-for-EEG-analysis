"""
Functions used for federated training.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
from collections import OrderedDict

try :
    import flwr
except :
    raise ImportError("To use the federated functions you need the flower framework. More info here https://pypi.org/project/flwr")

try :
    import wandb
except :
    raise ImportError("Failed import of wandb. The wandb_server class will not work.")

from library.model import hvEEGNet

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
# Flower framework client and server

class Client_V1(flwr.client.NumPyClient):
    def __init__(self, 
                 model : torch.nn.Module, 
                 train_loader : torch.utils.data.DataLoader, validation_loader : torch.utils.data.DataLoader, 
                 train_epoch_function, validation_epoch_function,
                 loss_function, optimizer, lr_scheduler,
                 train_config : dict
                 ) -> None :
        """
        Constructor of the flower client

        @param model : PyTorch model
        @param train_loader : PyTorch DataLoader used for training
        @param validation_loader : PyTorch DataLoader used for validation
        @param train_epoch_function : Function used for training
        @param validation_epoch_function : Function used for validation
        @param loss_function : Loss function used during training and validation
        @param optimizer : Optimizer used to train the model
        @param lr_scheduler : Learning rate scheduler
        @param train_config :Dictionary with the hyperparameter used for training
        """
        super().__init__()
        
        # Save dataloader used for training and validation
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        
        # Save training config
        self.train_config = train_config
        
        # Save model and move to training device
        self.model = model
        self.model.to(self.train_config["device"])

        # Save training and validation function
        self.train_epoch_function = train_epoch_function
        self.validation_epoch_function = validation_epoch_function

        # Save loss function, optimizer and learning rate scheduler
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        # ID used to save the client results in the server
        self.client_id = train_config['client_id']

    def get_parameters(self, config) :
        """
        Given a PyTorch model extract the weights/parameters, convert them in a numpy array and save them to a list

        @param model : PyTorch model

        @return weights_list : (list) List where each element is a group of weights of a specific layer/module of the Pytorch model
        """

        weights_list = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        return weights_list

    def get_properties(self, config) :
        return {}

    def set_weights(self, parameters_list : list):
        """
        Given a list of weights, each one represented as a numpy array, load the weights into the PyTorch model
        """
        # For each weight key (i.e. the name of parameters inside the model) save the key and the correspondent element in the parameters_list
        params_dict = zip(self.model.state_dict().keys(), parameters_list)

        # Convert the element in torch tensor
        # Note that from Python 3.7 normal dict works as OrderedDict. I use OrderedDict to keep compitability of the code with old python version
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        # Load the state dict
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters : list, config : dict) :
        """
        Train the provided parameters using the locally held dataset.
        """
        if self.train_config['print_var'] : print("START training client")

        # Update the model with the provided weights
        self.set_weights(parameters)
        
        # Variable used to save the metrics during the training
        train_loss_kl_list = []
        train_loss_recon_list = []

        # Dictionary with all the metrics of the client
        metrics_dict = {}

        # Epoch iteration
        for epoch in range(self.train_config["epochs"]) :
            if self.train_config['print_var'] : print("Epoch : {}".format(epoch))

            # Create log dict. It will be used to saved metrics and loss values after each epoch
            log_dict = {}

            # Advance epoch (TRAIN)
            _ = self.train_epoch_function( self.model, self.loss_function, self.optimizer, self.train_loader, self.train_config, log_dict = log_dict)

            # (OPTIONAL) Update learning rate
            if self.lr_scheduler is not None:
                # Save the current learning rate if I load the data on wandb
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

                # Update scheduler
                self.lr_scheduler.step()

            # Save metrics in the list
            train_loss_kl_list.append(log_dict['train_loss_kl'])
            train_loss_recon_list.append(log_dict['train_loss_recon'])

            # Saved metric in the dict (temporary workaround since flower not support a list as dictionary value)
            metrics_dict['train_loss_kl_{}'.format(epoch + 1)] = log_dict['train_loss_kl']
            metrics_dict['train_loss_recon_{}'.format(epoch + 1)] = log_dict['train_loss_recon']

        if self.train_config['print_var'] : print("END training client")

        # Saved total number of epoch
        metrics_dict['epochs'] = self.train_config['epochs']
        metrics_dict['client_id'] = self.client_id
        
        # Saved list with training loss per epoch (actually not supported by Flower)
        # metrics_dict['train_loss_kl_list'] = train_loss_kl_list
        # metrics_dict['train_loss_recon_list'] = train_loss_recon_list

        # Return variables, as specified in https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html#flwr.client.NumPyClient.fit
        return self.get_parameters(None), len(self.train_loader.dataset), metrics_dict

        # Give error when there is a list inside the dictionary
        # return self.get_parameters(None), len(self.train_loader.dataset), {"log_list" : log_list}

    def evaluate(self, parameters : list, config : dict) :
        """
        Evaluate the provided parameters using the locally held dataset.
        """

        # Update the model with the provided weights
        self.set_weights(parameters)
        
        # Create log dict. It will be used to saved metrics and loss values after each epoch
        log_dict = {}

        # Advance epoch (VALIDATION)
        validation_loss = self.validation_epoch_function( self.model, self.loss_function, self.validation_loader, self.train_config, log_dict = log_dict)
        
        # Save Client ID
        log_dict['client_id'] = self.client_id
        
        # Return variables, as specified in https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html#flwr.client.NumPyClient.evaluate
        return float(validation_loss), len(self.validation_loader.dataset), {}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Server function

class FedAvg_with_wandb(flwr.server.strategy.FedAvg):
    def __init__(self, server_config : dict()) :
        super().__init__()
        self.server_config = server_config 

        if 'notes' not in server_config['wandb_config'] : 
            server_config['wandb_config']['notest'] = 'No additional notes.'
        if 'name_training_run ' not in server_config['wandb_config'] or len(server_config['wandb_config']['name_training_run ']) == 0: 
            print('No name for the wandb run provided.')
            server_config['wandb_config']['name_training_run '] = None
        
        self.wandb_run = wandb.init(project = server_config['wandb_config']['project_name'], job_type = "train", config = server_config, notes = server_config['wandb_config']['notes'], name = server_config['wandb_config']['name_training_run '])
        self.wandb_run = None

        self.count_rounds = 0

    def aggregate_fit(self, server_round: int, results, failures) :
        """
        aggregate the results from the results from the clients and upload the results in wandb
        """

        self.count_rounds += 1

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # TODO Implements after tests

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays = flwr.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            # print(f"Saving round {server_round} aggregated_ndarrays...")
            # np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

            for i in range(len(results)) :
                metrics_from_training = results[i][1].metrics
                print(metrics_from_training)
            
            # TODO create model and load aggregated weights
            # torch.save(model.state_dict(), '{}/{}'.format(train_config['path_to_save_model'], "model_{}.pth".format(epoch + 1)))
        
            # TODO add check when the total number of round is reached 
            if self.count_rounds == None :
                # TODO terminate wandb run
                self.wandb_run = None 

        return aggregated_parameters, aggregated_metrics
