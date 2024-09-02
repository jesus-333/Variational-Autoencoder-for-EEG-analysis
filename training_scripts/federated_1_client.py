"""
Script to be used for the clients during federated training

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

from library.training import federated_training
from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic

from library.config import config_model as cm

import sys
import flwr
import numpy as np
import torch

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
        print("Path for the config file not specified as argument. Used the default path \"training_scripts/config/federated_client.toml\"")
        path_config_file = "training_scripts/config/federated_client.toml" 
    else :
        print("Path passed as argument.")
        path_config_file = sys.argv[1]

train_config = toml.load(path_config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_config['device'] = device

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get data (for now we used synthetic data to validate the code) and model

# Create synthetic data
train_data = np.random.rand(100, 1 , 3, 1000)
validation_data = np.random.rand(100, 1 , 3, 1000)

# Create channel lists
ch_list = ['C3', 'C5', 'C6']

# Create synthetic label
train_label = np.random.randint(0, 4, train_data.shape[0])
validation_label = np.random.randint(0, 4, validation_data.shape[0])

# Create train and validation dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, ch_list)

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# hvEEGNet creation
model = hvEEGNet.hvEEGNet_shallow(model_config)

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
train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
loader_list             = [train_dataloader, validation_dataloader]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Start client and train the model

server_address = "{}:{}".format(train_config['server_IP'], train_config['server_port'])

flwr.client.start_numpy_client(
    server_address = server_address,
    client = federated_training.Client_V1(
        model,
        train_dataloader, validation_dataloader,
        train_epoch_function, validation_epoch_function,
        loss_function, optimizer, lr_scheduler,
        train_config
    )
)
