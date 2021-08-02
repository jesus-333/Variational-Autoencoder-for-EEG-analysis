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
from support.support_training import advanceEpochV2, measureAccuracy
from support.support_datasets import Pytorch_Dataset_HGD
from support.support_visualization import visualizeHiddenSpace

#%% Settings

hidden_space_dimension = 64

print_var = True
tracking_input_dimension = True

epochs = 500
batch_size = 15
learning_rate = 1e-3
alpha = 0.5
repetition = 1

normalize_trials = True
early_stop = False
use_advance_vae_loss = False
measure_accuracy = True
save_model = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

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
        accuracy_train = []
        
        # Loss tracking variables (TEST)
        total_loss_test = []
        reconstruction_loss_test = []
        kl_loss_test = []
        discriminator_loss_test = []
        accuracy_test = []
        
        best_loss_test = sys.maxsize
        best_accuracy_test = sys.maxsize
        epoch_with_no_improvent = 0
        
        for epoch in range(epochs):
            # Training phase
            tmp_loss_train = advanceEpochV2(eeg_framework, device, train_dataloader, optimizer, is_train = True, use_advance_vae_loss = use_advance_vae_loss, alpha = alpha)
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
            # tmp_loss_test = advanceEpochV2(eeg_framework, device, test_dataloader, is_train = False, use_advance_vae_loss = use_advance_vae_loss, alpha = alpha)
            tmp_loss_test = [sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize]
            tmp_loss_test_total = tmp_loss_test[0]
            tmp_loss_test_recon = tmp_loss_test[1] 
            tmp_loss_test_kl = tmp_loss_test[2]
            tmp_loss_test_discriminator = tmp_loss_test[3]
            
            # Save tet losses
            total_loss_test.append(float(tmp_loss_test_total))
            reconstruction_loss_test.append(float(tmp_loss_test_recon))
            kl_loss_test.append(float(tmp_loss_test_kl))
            discriminator_loss_test.append(float(tmp_loss_test_discriminator))
            
            # Measure accuracy (OPTIONAL)
            if(measure_accuracy):
                # accuracy_train.append(measureAccuracy(eeg_framework.classifier, train_dataset))
                eeg_framework.eval()
                tmp_accuracy_test = measureAccuracy(eeg_framework.vae, eeg_framework.classifier, test_dataset, device)
                accuracy_test.append(tmp_accuracy_test)
                eeg_framework.train()
            
            if(tmp_loss_test_total < best_loss_test):
                # Update loss and accuracy (N.b. the best accuracy is intended as the accuracy when there is a new min loss)
                best_loss_test = tmp_loss_test_total
                best_accuracy_test = tmp_accuracy_test
                
                # Reset counter
                epoch_with_no_improvent = 0
                
                # visualizeHiddenSpace(eeg_framework.vae, trainlen_dataset, n_elements = 870, device = 'cuda')
                # visualizeHiddenSpace(eeg_framework.vae, test_dataset, True, n_elements = 159, device = 'cuda')
                
                # (OPTIONAL) Save the model
                if(save_model):  
                    save_path = "Saved model/eeg_framework"
                    if(use_advance_vae_loss): save_path = save_path + "_advance_loss"
                    else: save_path = save_path + "_normal_loss"
                    save_path = save_path + "_" + str(epoch) + ".pth"
                    
                    torch.save(eeg_framework.state_dict(), save_path)
   
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
                print("\t\tDiscriminator (TEST)\t: ", float(tmp_loss_test_discriminator))
                print("\t\tAccuracy (TEST)\t\t\t: ", float(tmp_accuracy_test), "\n")
                
                print("\tBest loss test\t\t: ", float(best_loss_test))
                print("\tBest Accuracy test\t: ", float(best_accuracy_test))
                print("\tNo Improvement\t\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break; 
       
# (OPTIONAL) Save the model
if(save_model):  
    save_path = "Saved model/eeg_framework"
    if(use_advance_vae_loss): save_path = save_path + "_advance_loss"
    else: save_path = save_path + "_normal_loss"
    save_path = save_path + "_" + str(epoch) + ".pth"
    
    torch.save(eeg_framework.state_dict(), save_path)
    
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

#%%

# eeg_framework.load_state_dict(torch.load("Saved model/eeg_framework_advance_loss_243.pth"))
idx_hidden_space = (31, 32)
    
# visualizeHiddenSpace(eeg_framework.vae, train_dataset, idx_hidden_space = idx_hidden_space, sampling = True, n_elements = 666, device = 'cuda')

visualizeHiddenSpace(eeg_framework.vae, train_dataset, idx_hidden_space = idx_hidden_space, sampling = False, n_elements = 666, device = 'cuda')
visualizeHiddenSpace(eeg_framework.vae, test_dataset, idx_hidden_space = idx_hidden_space, sampling = False, n_elements = 159, device = 'cuda')

#%%

if(False):
    # eeg_framework.load_state_dict(torch.load("Saved model/eeg_framework_normal_loss_389.pth"))
    train_accuracy = measureAccuracy(eeg_framework.vae, eeg_framework.classifier, train_dataset, device, True)
    test_accuracy = measureAccuracy(eeg_framework.vae, eeg_framework.classifier, test_dataset, device, True)
    
    print("TRAIN Accuracy: \t", train_accuracy)
    print("TEST accuracy: \t\t", test_accuracy)

#%%
if(False):
    # idx_list = [0,1,5,12,31,32,37,79,85]
    idx_list = [0,1,6,51,71,131,173,246,411,499]
    
    for idx in idx_list:
        eeg_framework.load_state_dict(torch.load("Saved model/eeg_framework_advance_loss_{}.pth".format(idx)))
        
        # visualizeHiddenSpace(eeg_framework.vae, test_dataset, False, n_elements = 159, device = 'cuda')
        
        test_accuracy_evaluated = measureAccuracy(eeg_framework.vae, eeg_framework.classifier, test_dataset, device, False)
        test_accuracy_evaluated = round(test_accuracy_evaluated * 100, 2)
        test_accuracy_saved = round(accuracy_test[idx] * 100, 2)
        
        print("Epoch: ", idx)
        print("\tTEST accuracy eval:\t\t", test_accuracy_evaluated)
        print("\tTEST accuracy saved:\t",test_accuracy_saved)