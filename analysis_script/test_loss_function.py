"""
Test loss function and backpropagation with synthetic data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

from library.model import hvEEGNet
from library.dataset import dataset_time as ds_time
from library.training import train_generic
from library.config import config_model as cm, config_training as ct

torch.cuda.empty_cache()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

device = 'cpu'
# device = 'cuda'

C = 22
T = 1000
batch_size = 8

gamma_dtw = 1

delete_variables = False
np.random.seed(43)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create synthetic data

# Create data
train_data = torch.rand(3 * batch_size, 1, C, T).numpy()
train_label : np.ndarray = np.random.randint(0, 4, train_data.shape[0])
ch_list = torch.rand(C)

# Create dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model and training parameters

# Model
model = hvEEGNet.hvEEGNet_shallow(cm.get_config_hierarchical_vEEGNet(C, T)) # new model is instantiated for each iteration of the loop.

# Train config
train_config = ct.get_config_hierarchical_vEEGNet_training()
train_config['device'] = device
train_config['gamma_dtw'] = gamma_dtw

# Loss function
loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = train_config['lr'], weight_decay = train_config['optimizer_weight_decay'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test loss function

# Move model to train device
model.to(train_config['device'])

# Get a random batch of data
idx = np.random.randint(low = 0, high = len(train_dataset) - batch_size)
sample_batch, _ = train_dataset[idx:idx + batch_size]
x = sample_batch.to(train_config['device'])

# Forward pass
x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)
true_label = None
predict_label = None

# Loss evaluation
batch_train_loss = loss_function.compute_loss(x, x_r,
                                         mu_list, log_var_list,
                                         delta_mu_list, delta_log_var_list,
                                         predict_label, true_label)

# Get the various losses 
train_loss = batch_train_loss[0] * x.shape[0]
recon_loss = batch_train_loss[1] * x.shape[0]
kl_loss    = batch_train_loss[2] * x.shape[0]

# Print the results
str_to_print  = "Train loss total : {}\n".format(train_loss)
str_to_print += "Train loss recon : {}\n".format(recon_loss)
str_to_print += "Train loss KL    : {}\n".format(kl_loss)
print(str_to_print)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if delete_variables :
    del x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, batch_train_loss
    gc.collect()

