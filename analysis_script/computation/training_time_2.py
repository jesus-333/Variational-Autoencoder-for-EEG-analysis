"""
Used synthetic data to compute the average duration of a training epoch for hvEEGNet
The time is computed for a list of values for dataset and batch size
The input size remian fix.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports the library modules

import torch
import numpy as np
import time
import os

from library.model import hvEEGNet
from library.dataset import dataset_time as ds_time
from library.training import train_generic

from library.config import config_model as cm
from library.config import config_training as ct

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_training_time(model, loss_function, optimizer, dataloader, train_config) :
    time_list = []

    for rep in range(epoch_repetition) :
        print("\t{} repetition".format(rep + 1))

        start = time.time()
        _ = train_epoch_function(model, loss_function, optimizer, dataloader, train_config, None)
        time_list.append(time.time() - start)
    
    return time_list

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# Input size
C = 22      # Number of EEG channels
T = 1000    # Number of time samples

# Synthetic dataset parameters
n_elements_in_the_dataset_list = [10, 50, 100, 150, 200]
n_elements_in_the_dataset_list = [100, 150, 200]
batch_size_list = [5, 10, 15]
batch_size_list = [5, 10]

# Number of time the training epoch is repeated to compute the average time
epoch_repetition = 5

# Training device (cpu/gpu)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# If True save the matrix in a npy file
save_results = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Stuff used during the computations

# Model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Training config
train_config = ct.get_config_hierarchical_vEEGNet_training()
train_config['device'] = device

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)

# Move model to training device
model.to(device)

# Matrix to save the results
training_time_matrix_mean = np.zeros((len(n_elements_in_the_dataset_list), len(batch_size_list)))
training_time_matrix_std = np.zeros((len(n_elements_in_the_dataset_list), len(batch_size_list)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute training time

for i in range(len(n_elements_in_the_dataset_list)) :
    # Get number of elements in the dataset
    n_elements_in_the_dataset = n_elements_in_the_dataset_list[i]
    
    for j in range(len(batch_size_list)) :
        # Get batch size
        batch_size = batch_size_list[j]
        
        # Set batch size in the training config
        train_config['batch_size'] = batch_size

        print("Computation of n. elements dataset = {} and batch size = {}".format(n_elements_in_the_dataset, batch_size))

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

        time_list = compute_training_time(model, loss_function, optimizer, train_dataloader, train_config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Save results

        training_time_matrix_mean[i, j] = np.mean(time_list)
        training_time_matrix_std[i, j] = np.std(time_list)

        if save_results :
            # Save path and file name
            file_name = 'time_list'
            path_save = 'Saved Results/computation time/n_elements_{}_batch_{}/'.format(n_elements_in_the_dataset, batch_size)

            # Create the path if it does not exist
            os.makedirs(path_save, exist_ok = True)
            
            # Save matrix in npy format
            np.save(path_save + file_name + '.npy', time_list)

            # Save matrix in text format
            np.savetxt(path_save + file_name + '.txt', time_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print results

if save_results : print("Note that the saved results are only the lists of time for each combinations of n. elements/batch size. It is recommended to add a txt file with hardware specifications, epoch repetition etc")

for i in range(len(n_elements_in_the_dataset_list)) :
    n_elements_in_the_dataset = n_elements_in_the_dataset_list[i]
    for j in range(len(batch_size_list)) :
        batch_size = batch_size_list[j]
        print("Dataset size = {}, Batch size = {}\t Training time = {:.2f}s ± {:.2f}s".format(n_elements_in_the_dataset, batch_size, training_time_matrix_mean[i, j], training_time_matrix_std[i, j]))
