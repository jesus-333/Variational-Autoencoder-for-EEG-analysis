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

from support.VAE_EEGNet import EEGFramework
from support.support_training import VAE_loss, advanceEpochV2, Pytorch_Dataset_HGD

#%% Settings

hidden_space_dimension = 2

print_var = True
tracking_input_dimension = True

epochs = 500
batch_size = 15
learning_rate = 1e-3
alpha = 1
repetition = 1

normalize_trials = True
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
        train_dataset = Pytorch_Dataset_HGD(path, normalize_trials = normalize_trials)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TRAIN dataset and dataloader created\n")
        
        # Test dataset
        path = 'Dataset/HGD/Test/{}/'.format(idx)
        test_dataset = Pytorch_Dataset_HGD(path, normalize_trials = normalize_trials)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TEST dataset and dataloader created\n")
        
        # Network creation
        C = train_dataset[0][0].shape[1]
        T = train_dataset[0][0].shape[2]
        eeg_framework = EEGFramework(C = C, T = T, hidden_space_dimension = hidden_space_dimension, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        if(print_var): print("VAE created")
        
        # Optimizer
        # optimizer = torch.optim.Adam(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        optimizer = torch.optim.AdamW(eeg_framework.parameters(), lr = learning_rate, weight_decay = 1e-5)
        # optimizer = torch.optim.SGD(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        
        # Loss tracking variables (TRAIN)
        total_loss_train = []
        reconstruction_loss_train = []
        kl_loss_train = []
        discriminator_loss_train = []
        
         # Loss tracking variables (TEST)
        total_loss_test = []
        reconstruction_loss_test = []
        kl_loss_test = []
        discriminator_loss_test = []
        
        best_loss_test = sys.maxsize
        
        for epoch in range(epochs):
            # Training phase
            tmp_loss_train = advanceEpochV2(eeg_framework, device, train_dataloader, optimizer, is_train = True, alpha = alpha)
            tmp_loss_train_total = tmp_loss_train[0]
            tmp_loss_train_recon = tmp_loss_train[1]
            tmp_loss_train_kl = tmp_loss_train[2]
            tmp_loss_train_discriminator = tmp_loss_train[3]
            
            # Save train losses
            total_loss_train.append(float(tmp_loss_train_total))
            reconstruction_loss_train.append(float(tmp_loss_train_recon))
            kl_loss_train.append(float(tmp_loss_train_kl))
            discriminator_loss_train.append(float(tmp_loss_train_discriminator))
            
            # Testing phase
            tmp_loss_test = advanceEpochV2(eeg_framework, device, test_dataloader, is_train = False, alpha = alpha)
            tmp_loss_test_total = tmp_loss_test[0]
            tmp_loss_test_recon = tmp_loss_test[1] 
            tmp_loss_test_kl = tmp_loss_test[2]
            tmp_loss_test_discriminator = tmp_loss_test[3]
            
            # Save tet losses
            total_loss_test.append(float(tmp_loss_test_total))
            reconstruction_loss_test.append(float(tmp_loss_test_recon))
            kl_loss_test.append(float(tmp_loss_test_kl))
            discriminator_loss_test.append(float(tmp_loss_test_discriminator))
            
            if(tmp_loss_test_total < best_loss_test):
                # Update loss
                best_loss_test = tmp_loss_test_total
                
                # Reset counter
                epoch_with_no_improvent = 0
            else: 
                epoch_with_no_improvent += 1
            
            if(print_var and epoch % step_show == 0):
                print("Epoch: {} ({:.2f}%) - Subject: {} - Repetition: {}".format(epoch, epoch/epochs * 100, idx, rep))
                
                print("\tLoss (TRAIN)\t: ", float(tmp_loss_train_total))
                print("\t\tReconstr (TRAIN)\t\t: ", float(tmp_loss_train_recon))
                print("\t\tKullback (TRAIN)\t\t: ", float(tmp_loss_train_kl))
                print("\t\tDiscriminator (TRAIN)\t: ", float(tmp_loss_train_discriminator), "\n")
                
                print("\tLoss (TEST)\t\t: ", float(tmp_loss_test_total))
                print("\t\tReconstr (TEST)\t\t\t: ", float(tmp_loss_test_recon))
                print("\t\tKullback (TEST)\t\t\t: ", float(tmp_loss_test_kl))
                print("\t\tDiscriminator (TEST)\t: ", float(tmp_loss_test_discriminator), "\n")
                
                print("\tBest loss test\t: ", float(best_loss_test))
                print("\tNo Improvement\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break; 
                
#%%

plt.figure()
plt.plot(total_loss_train)
plt.plot(total_loss_test)
plt.legend(["Train", "Test"])
plt.title("Total Loss")

plt.figure()
plt.plot(reconstruction_loss_train)
plt.plot(reconstruction_loss_test)
plt.legend(["Train", "Test"])
plt.title("Reconstruction Loss")

plt.figure()
plt.plot(kl_loss_train)
plt.plot(kl_loss_test)
plt.legend(["Train", "Test"])
plt.title("KL Loss")

plt.figure()
plt.plot(discriminator_loss_train)
plt.plot(discriminator_loss_test)
plt.legend(["Train", "Test"])
plt.title("Discriminator LOSS")