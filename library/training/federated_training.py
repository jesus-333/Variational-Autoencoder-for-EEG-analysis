"""
Functions to used for federated training on edge device
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
from collections import OrderedDict

try :
    import flwr
except :
    raise ImportError("To use the federated functions you need the flower framework. More info here https://pypi.org/project/flwr")

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

def set_weights(model : torch.nn.Module, parameters_list : list):
    """
    Given a list of weights, each one represented as a numpy array, load the weights into the PyTorch model
    """
    # For each weight key (i.e. the name of parameters inside the model) save the key and the correspondent element in the parameters_list
    params_dict = zip(model.state_dict().keys(), parameters_list)

    # Convert the element in torch tensor
    # Note that from Python 3.7 normal dict works as OrderedDict. I use OrderedDict to keep compitability of the code with old python version
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Load the state dict
    model.load_state_dict(state_dict, strict=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Flower framework client and server

class FlowerClient(flwr.client.NumPyClient):
    def __init__(self, 
                 model : torch.nn.Module, 
                 train_loader : torch.utils.data.DataLoader, validation_loader : torch.utils.data.DataLoader, 
                 train_epoch_function, validation_epoch_function,
                 loss_function, optimizer,
                 train_config : dict
                 ):
        """
        Constructor of the flower client

        @param model : PyTorch model
        @param train_loader : PyTorch DataLoader used for training
        @param validation_loader : PyTorch DataLoader used for validation
        @param train_epoch_function : Function used for training
        @param validation_epoch_function : Function used for validation
        @param loss_function : Loss function used during training and validation
        @param optimizer : Optimizer used to train the model
        @param train_config :Dictionary with the hyperparameter used for training
        """
        
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

        # Save loss function and optimizer
        self.loss_function = loss_function
        self.optimizer = optimizer

    def fit(self, parameters : list, config : dict) :
        """
        Train the provided parameters using the locally held dataset.
        """

        # Update the model with the provided weights
        set_weights(self.model, parameters)
        
        # List used to save the results of each epoch
        log_list = []

        # Epoch iteration
        for epoch in self.train_config["epochs"] :
            # Create log dict. It will be used to saved metrics and loss values after each epoch
            log_dict = {}

            # Advance epoch (TRAIN)
            _ = self.train_epoch_function( self.model, self.loss_function, self.optimizer, self.train_loader, self.train_config, log_dict = log_dict)
            
            # Save log in the list
            log_list.append(log_dict)

        # Return variables, as specified in https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html#flwr.client.NumPyClient.fit
        return get_weights(self.net), len(self.trainloader.dataset), log_dict

    def evaluate(self, parameters : list, config : dict) :
        """
        Evaluate the provided parameters using the locally held dataset.
        """

        # Update the model with the provided weights
        set_weights(self.model, parameters)
        
        # List used to save the results of each epoch
        log_list = []

        # Epoch iteration
        for epoch in self.train_config["epochs"] :
            # Create log dict. It will be used to saved metrics and loss values after each epoch
            log_dict = {}

            # Advance epoch (VALIDATION)
            validation_loss = self.validation_epoch_function( self.model, self.loss_function, self.optimizer, self.train_loader, self.train_config, log_dict = log_dict)
            
            # Save log in the list
            log_list.append(log_dict)
        
        # Return variables, as specified in https://flower.ai/docs/framework/ref-api/flwr.client.NumPyClient.html#flwr.client.NumPyClient.evaluate
        return validation_loss, len(self.validation_loader.dataset), log_dict

