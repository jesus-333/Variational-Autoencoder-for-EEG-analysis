"""
Function used to simulate federated training if multiple devices are not available
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch

from . import client
from ...model import hvEEGNet
from ...training import train_generic

try :
    import flwr
except :
    raise ImportError("To use the federated functions you need the flower framework. More info here https://pypi.org/project/flwr")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def generate_client_function_hvEEGNet_training(model_config_list : list, train_config_list : list,
                                               train_dataset_list : list, validation_dataset_list : list
                                               ) :
    """
    This function return a function that create the clien based on id. The function will be used by the VirtualClientEngine to spawn a client with client id 'cid'
    (More info around min 26 of this video https://www.youtube.com/watch?v=QR3BincFnIw)
    The simulations will used the client id to get the corresponding parameters, datasets, etc from the list provided in input.
    Then the i-th element of each list will be the ones used for the corresponding client.

    E.g. if I have 9 clients, each one of the input lists must have 9 element.
         The i-th element of each list corresponds to the datasets for that client

    @param model_config_list : (list) List with the model configs for each client
    @param train_config_list : (list) List with the train configs for each client
    @param train_dataset_list : (list) List with the training dataset for each client
    @param validation_dataset_list : (list) List with the validation dataset for each client
    """

    if len(train_dataset_list) != len(validation_dataset_list) :
        raise ValueError("train_dataset_list and validation_dataset_list must contains the same number of dataset (i.e. have the same length)\nCurrent length :\n\ttrain_dataset_list : {}\n\tvalidation_dataset_list : {}".format(len(train_dataset_list), len(validation_dataset_list)))


    def client_function_hvEEGNet_training(cid : str) :
        """
        This function generate a clien with the corresponding client id.

        @param cid : (str) Client id
        """

        # Convert id from string to int
        id = int(cid)

        # Get configs for the specifi client
        model_config = model_config_list[id]
        train_config = train_config_list[id]

        # If the model has also a classifier add the information to training config
        train_config['measure_metrics_during_training'] = model_config['use_classifier']
        train_config['use_classifier'] = model_config['use_classifier']

        # hvEEGNet creation
        model = hvEEGNet.hvEEGNet_shallow(model_config_list[id])

        # Get training and validaiton function
        train_epoch_function, validation_epoch_function = train_generic.get_train_and_validation_function(model)

        # Get loss function for hvEEGNet
        loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = train_config['lr'],
                                      weight_decay = train_config['optimizer_weight_decay']
                                      )

        # (OPTIONAL) Setup lr scheduler
        if train_config['use_scheduler'] :
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
        else:
            lr_scheduler = None

        # Create dataloader
        train_dataloader        = torch.utils.data.DataLoader(train_dataset_list[id], batch_size = train_config['batch_size'], shuffle = True)
        validation_dataloader   = torch.utils.data.DataLoader(validation_dataset_list[id], batch_size = train_config['batch_size'], shuffle = True)

        tmp_client = client.Client_V1(
            model,
            train_dataloader, validation_dataloader,
            train_epoch_function, validation_epoch_function,
            loss_function, optimizer, lr_scheduler,
            train_config
        )

        return tmp_client

    return client_function_hvEEGNet_training

