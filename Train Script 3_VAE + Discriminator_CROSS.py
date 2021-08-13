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
from scipy.io import loadmat, savemat

import torch
from torch.utils.data import DataLoader

from support.VAE_EEGNet import EEGFramework
from support.support_training import advanceEpochV2, measureAccuracy, measureSingleSubjectAccuracy
from support.support_datasets import PytorchDatasetEEGSingleSubject, PytorchDatasetEEGMergeSubject
from support.support_visualization import visualizeHiddenSpace

#%% Settings

dataset_type = 'D2A'

hidden_space_dimension = 64

print_var = True
tracking_input_dimension = True

epochs = 500
batch_size = 15
learning_rate = 1e-3
alpha = 0.01 #TODO REMOVE THIS
repetition = 2  

normalize_trials = False
use_reparametrization = False 
merge_subject = True
execute_test_epoch = True
early_stop = False
use_advance_vae_loss = False
measure_accuracy = True
save_model = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [2, 4, 7]

step_show = 2

#%% Variable for saving results

accuracy_test_list = []
best_average_accuracy_for_repetition = []
subject_accuracy_during_epochs_LOSS = []
subject_accuracy_during_epochs_for_repetition_LOSS = []
best_subject_accuracy_for_repetition_END = np.zeros((9, repetition))
best_subject_accuracy_for_repetition_LOSS = np.zeros((9, repetition))

#%% Training cycle
for rep in range(repetition):
    for idx in idx_list:
        
        subject_accuracy_during_epochs_LOSS = []
        
        merge_list = idx_list.copy()
        merge_list.remove(idx)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Train dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Train/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Train/{}/'.format(idx)
            
        train_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = merge_list)
            
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TRAIN dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Test dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Test/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Test/{}/'.format(idx)
        
        test_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = [idx])
            
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TEST dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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
            if(execute_test_epoch): tmp_loss_test = advanceEpochV2(eeg_framework, device, test_dataloader, is_train = False, use_advance_vae_loss = use_advance_vae_loss, alpha = alpha)
            else: tmp_loss_test = [sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize]
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
                tmp_accuracy_test = measureAccuracy(eeg_framework.vae, eeg_framework.classifier, test_dataset, device, use_reparametrization)
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
                
                if(measure_accuracy):
                    tmp_subject_accuracy = measureSingleSubjectAccuracy(eeg_framework, [idx], dataset_type, use_reparametrization, device = device)[0]
                    subject_accuracy_during_epochs_LOSS.append(tmp_subject_accuracy)
                    best_subject_accuracy_for_repetition_LOSS[idx - 1, rep] = tmp_subject_accuracy
                
                # (OPTIONAL) Save the model
                if(save_model):  
                    save_path = "Saved model/eeg_framework"
                    if(use_advance_vae_loss): save_path = save_path + "_advance_loss"
                    else: save_path = save_path + "_normal_loss"
                    save_path = save_path + "_" + str(rep)
                    save_path = save_path + "_" + str(epoch) + ".pth"
                    
                    torch.save(eeg_framework.state_dict(), save_path)
                    
                    if(print_var): print("SAVED MODEL AT EPOCH: ", epoch)
   
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
                print("\tBest Accuracy test (1)\t: ", float(best_accuracy_test))
                print("\tBest Accuracy test (2)\t: ", float(np.max(accuracy_test)))
                print("\tNo Improvement\t\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break; 
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        best_subject_accuracy_for_repetition_END[idx - 1, rep] = measureSingleSubjectAccuracy(eeg_framework, [idx], dataset_type, use_reparametrization, device = device)[0]
        # best_subject_accuracy_for_repetition_LOSS[idx - 1, rep] = subject_accuracy_during_epochs_LOSS[-1]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # (OPTIONAL) Save the model
    if(save_model):  
        save_path_END = "Saved model/eeg_framework"
        if(use_advance_vae_loss): save_path_END = save_path_END + "_advance_loss"
        else: save_path_END = save_path_END + "_normal_loss"
        save_path_END = save_path_END + "_" + str(epoch) + ".pth"
        
        torch.save(eeg_framework.state_dict(), save_path_END)
    
    
   
