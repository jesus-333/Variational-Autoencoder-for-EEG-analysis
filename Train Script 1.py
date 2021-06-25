"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%% Path for imports

import sys
sys.path.insert(1, 'support')

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
from torch.utils.data import DataLoader

from support.VAE_EEGNet import EEGNetVAE
from support.support_training import VAE_loss, advanceEpoch, Pytorch_Dataset_HGD

#%% Settings

hidden_space_dimension = 4

print_var = True
tracking_input_dimension = True

epochs = 500
batch_size = 15
learning_rate = 1e-3
alpha = 1
repetition = 1

early_stop = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

loss_fn = VAE_loss

idx_list = [3]

step_show = 2

#%% Training cycle

for rep in range(repetition):
    for idx in idx_list:
        
        # Train dataset
        path = 'Dataset/HGD/Train/{}/'.format(idx)
        train_dataset = Pytorch_Dataset_HGD(path)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TRAIN dataset and dataloader created\n")
        
        # Test dataset
        path = 'Dataset/HGD/Test/{}/'.format(idx)
        test_dataset = Pytorch_Dataset_HGD(path)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TEST dataset and dataloader created\n")
        
        # Network creation
        C = train_dataset[0][0].shape[1]
        T = train_dataset[0][0].shape[2]
        vae = EEGNetVAE(C = C, T = T, hidden_space_dimension = hidden_space_dimension, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        if(print_var): print("VAE created")
        
        # Optimizer
        # optimizer = torch.optim.Adam(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        optimizer = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        # optimizer = torch.optim.SGD(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        
        # Loss tracking variable
        total_loss_train = []
        total_loss_test = []
        best_loss_test = sys.maxsize
        
        for epoch in range(epochs):
            # Training phase
            tmp_loss_train = advanceEpoch(vae, device, train_dataloader, loss_fn, optimizer, is_train = True, alpha = alpha)
            total_loss_train.append(float(tmp_loss_train))
            
            # Testing phase
            tmp_loss_test = advanceEpoch(vae, device, test_dataloader, loss_fn, is_train = False, alpha = alpha)
            total_loss_test.append(float(tmp_loss_test))
            
            if(tmp_loss_test < best_loss_test):
                # Update loss
                best_loss_test = tmp_loss_test
                
                # Reset counter
                epoch_with_no_improvent = 0
            else: 
                epoch_with_no_improvent += 1
            
            if(print_var and epoch % step_show == 0):
                print("Epoch: {} ({:.2f}%) - Subject: {} - Repetition: {}".format(epoch, epoch/epochs * 100, idx, rep))
                print("     Loss (TRAIN)\t: ", float(tmp_loss_train))
                print("     Loss (TEST)\t: ", float(tmp_loss_test))
                print("     Best loss test\t: ", float(best_loss_test))
                print("     No Improvement\t: ", int(epoch_with_no_improvent))
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break;
