"""
Used synthetic data to compute the average duration of a training epoch for hvEEGNet
The time is computed for a specific dataset and batch size and for a specific input size.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports the library modules

import torch
import numpy as np
import time

from library.model import hvEEGNet
from library.dataset import dataset_time as ds_time
from library.training import train_generic

from library.config import config_model as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# Input size
C = 22      # Number of EEG channels
T = 1000    # Number of time samples

# Dataset parameters
n_elements_in_the_dataset = 100
batch_size = 5

# Number of time the training epoch is repeated to compute the average time
epoch_repetition = 10

# Training device (cpu/gpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create model and get dictionaries with configs

# Model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Training config
train_config = ct.get_config_hierarchical_vEEGNet_training()
train_config['batch_size'] = batch_size
train_config['device'] = device

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)

# Move model to training device
model.to(device)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create synthetic data and Dataloader

# Create synthetic data
train_data = np.random.rand(n_elements_in_the_dataset, 1 , C, T)
train_label = np.random.randint(0, 4, train_data.shape[0])

# Create train dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list = None)

# Create dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get functions and other stuff used for training

# Training function
train_epoch_function, _ = train_generic.get_train_and_validation_function(model)

# loss function and optimizer
loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)
optimizer = torch.optim.AdamW(model.parameters(), lr = train_config['lr'], weight_decay = train_config['optimizer_weight_decay'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute average training time for epoch

time_list = []

for i in range(epoch_repetition) :
    print("{} repetition".format(i + 1))

    start = time.time()
    _ = train_epoch_function(model, loss_function, optimizer, train_dataloader, train_config, None)
    time_list.append(time.time() - start)
    
print("The average training time for an epoch is {:.2f}s ± {:.2f}s".format(np.mean(time_list), np.std(time_list)))
