"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the VAE and Discriminator JOINTLY for INTRA subject classification.
The train datasets are merged toghether and then the network is tested on the test datasets separately.

"""

#%% Path for imports

import sys
sys.path.insert(0, 'support')

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

import torch
from torch.utils.data import DataLoader

from support.VAE_EEGNet import EEGFramework
from support.support_training import advanceEpochV2, measureAccuracyAndKappaScore, measureSingleSubjectAccuracyAndKappaScore
from support.support_datasets import PytorchDatasetEEGSingleSubject, PytorchDatasetEEGMergeSubject
from support.support_visualization import visualizeHiddenSpace

#%% Settings

dataset_type = 'D2A'

hidden_space_dimension = 64
# hidden_space_dimension = 32 # TODO Remove

print_var = True
tracking_input_dimension = True

# Training parameters
epochs = 500
batch_size = 15
learning_rate = 1e-3
repetition = 30
percentage_train = 0.9

# Hyperparameter for the loss function (alpha for recon, beta for kl, gamma for classifier)
alpha = 0.1
beta = 1
gamma = 1

# Various parameter
normalize_trials = True # TODO check if decoder use sigmoid as last layer
execute_test_epoch = True
early_stop = False
use_shifted_VAE_loss = False
use_reparametrization_for_classification = False
measure_accuracy = True
save_model = False
plot_reconstructed_signal = False
L2_loss_type = 0
use_lr_scheduler = False
optimize_memory_dataset = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')

idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [2, 4, 7]

step_show = 2

#%% Variable for saving results

# Accuracy variable
accuracy_test_list = []
best_average_accuracy_for_repetition = []
subject_accuracy_during_epochs_LOSS = []
subject_accuracy_during_epochs_for_repetition_LOSS = []
best_subject_accuracy_for_repetition_END = np.zeros((9, repetition))
best_subject_accuracy_for_repetition_LOSS = np.zeros((9, repetition))

# Kappa Score variable
subject_kappa_score_during_epochs_LOSS = []
subject_kappa_score_during_epochs_for_repetition_LOSS = []
best_subject_kappa_score_for_repetition_END = np.zeros((9, repetition))
best_subject_kappa_score_for_repetition_LOSS = np.zeros((9, repetition))

parameter_train = []

#%% Check on parameter

if(repetition < 0): raise Exception("The number of repetition must be greater than 0")
if(L2_loss_type != 0 and L2_loss_type != 1 and L2_loss_type != 2): raise Exception("L2_loss_type must be 0 or 1 or 2") 
if(alpha < 0 or beta < 0 or gamma < 0): raise Exception("The loss hyperparameter (alpha, beta, gamma) must be greater than 0 ") 

#%% Training cycle


merge_list = idx_list.copy()
idx_list = [1]

# TODO REMOVE
tmp_weight_decay = 1e-5
    

for rep in range(repetition):
    
    if(rep >= 5): tmp_weight_decay = 1e-2;
    if(rep >= 10): tmp_weight_decay = 1e-5; hidden_space = 32
    if(rep >= 15): tmp_weight_decay = 1e-2;
    if(rep >= 20): tmp_weight_decay = 1e-5; hidden_space = 16
    if(rep >= 25): tmp_weight_decay = 1e-2;
    
    tmp_parameter_train = {'lr':learning_rate, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 
                       'use_lr_scheduler':use_lr_scheduler, 'L2_loss_type': L2_loss_type, 'epochs':epochs,
                       'hidden_space_dimension':hidden_space_dimension
                       }
    
    for idx in idx_list:
        
        # Reset list 
        subject_accuracy_during_epochs_LOSS = []
        subject_kappa_score_during_epochs_LOSS = []
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Train dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Train/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Train/{}/'.format(idx)
        
        full_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = merge_list, normalize_trials = normalize_trials, optimize_memory = optimize_memory_dataset, device = device)
        
        size_train = int(len(full_dataset) * percentage_train) 
        size_val = len(full_dataset) - size_train
        train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [size_train, size_val])
        
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TRAIN dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Test dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Test/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Test/{}/'.format(idx)
        
        test_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = merge_list, normalize_trials = normalize_trials, optimize_memory = optimize_memory_dataset, device = device)
        
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TEST dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Network creation
        C = train_dataset[0][0].shape[1]
        T = train_dataset[0][0].shape[2]
        eeg_framework = EEGFramework(C = C, T = T, hidden_space_dimension = hidden_space_dimension, use_reparametrization_for_classification = use_reparametrization_for_classification, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        if(print_var): print("EEG Framework created")
        
        # Optimizer
        # optimizer = torch.optim.AdamW(eeg_framework.parameters(), lr = learning_rate, weight_decay = 1e-5)
        optimizer = torch.optim.AdamW(eeg_framework.parameters(), lr = learning_rate, weight_decay = tmp_weight_decay)
        
        # Learning weight scheduler 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 0)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Tracking variable (for repetition)
        
        # Loss tracking variables (TRAIN)
        total_loss_train = []
        reconstruction_loss_train = []
        kl_loss_train = []
        discriminator_loss_train = []
        
        # Loss tracking variables (VALIDATION)
        total_loss_val = []
        reconstruction_loss_val = []
        kl_loss_val = []
        discriminator_loss_val = []

        # Loss tracking variables (TEST)
        total_loss_test = []
        reconstruction_loss_test = []
        kl_loss_test = []
        discriminator_loss_test = []
        accuracy_test = []
        kappa_score_test = []
        
        best_loss_validation = sys.maxsize
        
        best_accuracy_test = sys.maxsize
        best_kappa_score_test = sys.maxsize
        epoch_with_no_improvent = 0
        
        for epoch in range(epochs):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Training phase
            tmp_loss_train = advanceEpochV2(eeg_framework, device, train_dataloader, optimizer, 
                                            is_train = True, use_shifted_VAE_loss = use_shifted_VAE_loss, 
                                            alpha = alpha, beta = beta, gamma = gamma,
                                            L2_loss_type = L2_loss_type)
            tmp_loss_train_total = tmp_loss_train[0]
            tmp_loss_train_recon = tmp_loss_train[1]
            tmp_loss_train_kl = tmp_loss_train[2]
            tmp_loss_train_discriminator = tmp_loss_train[3]
            
            # Save train losses
            total_loss_train.append(float(tmp_loss_train_total))
            reconstruction_loss_train.append(float(tmp_loss_train_recon))
            kl_loss_train.append(float(tmp_loss_train_kl))
            discriminator_loss_train.append(float(tmp_loss_train_discriminator))
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Validation phase
            tmp_loss_val = advanceEpochV2(eeg_framework, device, validation_dataloader, optimizer, 
                                            is_train = False, use_shifted_VAE_loss = use_shifted_VAE_loss, 
                                            alpha = alpha, beta = beta, gamma = gamma,
                                            L2_loss_type = L2_loss_type)
            tmp_loss_val_total = tmp_loss_val[0]
            tmp_loss_val_recon = tmp_loss_val[1]
            tmp_loss_val_kl = tmp_loss_val[2]
            tmp_loss_val_discriminator = tmp_loss_val[3]
            
            # Save train losses
            total_loss_train.append(float(tmp_loss_train_total))
            reconstruction_loss_train.append(float(tmp_loss_train_recon))
            kl_loss_train.append(float(tmp_loss_train_kl))
            discriminator_loss_train.append(float(tmp_loss_train_discriminator))
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Testing phase
            if(execute_test_epoch): tmp_loss_test = advanceEpochV2(eeg_framework, device, test_dataloader, 
                                                                   is_train = False, use_shifted_VAE_loss = use_shifted_VAE_loss, 
                                                                   alpha = alpha, beta = beta, gamma = gamma,
                                                                   L2_loss_type = L2_loss_type)
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
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Measure accuracy (OPTIONAL)
            if(measure_accuracy):
                # Switch to evaluation mode
                eeg_framework.eval()
                
                # Evaluate accuracy and kappa score
                tmp_accuracy_test, tmp_kappa_score_test = measureAccuracyAndKappaScore(eeg_framework.vae, eeg_framework.classifier, test_dataset, device, use_reparametrization_for_classification)
                
                # Save results
                accuracy_test.append(tmp_accuracy_test)
                kappa_score_test.append(tmp_kappa_score_test)
                
                # Return to train mode
                eeg_framework.train()
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Check loss value
            
            if(tmp_loss_val_total < best_loss_validation):
                # Update loss and (OPTIONAL) accuracy (N.b. the best accuracy is intended as the accuracy when there is a new min loss)
                best_loss_validation = tmp_loss_val_total
                if(measure_accuracy):
                    best_accuracy_test = tmp_accuracy_test
                    best_kappa_score_test = tmp_kappa_score_test
                
                # Reset counter
                epoch_with_no_improvent = 0
                
                # Measure the accuracy separately for each subject
                if(measure_accuracy):
                    tmp_subject_accuracy, tmp_subject_kappa_score = measureSingleSubjectAccuracyAndKappaScore(eeg_framework, merge_list, dataset_type, normalize_trials = normalize_trials, use_reparametrization_for_classification = use_reparametrization_for_classification, device = device)
                    subject_accuracy_during_epochs_LOSS.append(tmp_subject_accuracy)
                    subject_kappa_score_during_epochs_LOSS.append(tmp_subject_kappa_score)
                    
                if(plot_reconstructed_signal):
                    x = test_dataset[np.random.randint(0, len(train_dataset))][0]
                    x_r = eeg_framework(x.unsqueeze(0).cuda())[0].cpu().squeeze().detach()
                    
                    plt.figure(figsize = (20, 10))
                    plt.plot(x.squeeze()[0])
                    plt.plot(x_r.squeeze()[0])
                    plt.show()
                
                # (OPTIONAL) Save the model
                if(save_model):  
                    save_path = "Saved model/eeg_framework"
                    if(use_shifted_VAE_loss): save_path = save_path + "_advance_loss"
                    else: save_path = save_path + "_normal_loss"
                    save_path = save_path + "_" + str(rep)
                    save_path = save_path + "_" + str(epoch) + ".pth"
                    
                    torch.save(eeg_framework.state_dict(), save_path)
                    
                    if(print_var): print("SAVED MODEL AT EPOCH: ", epoch)
   
            else: 
                epoch_with_no_improvent += 1
            
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # (OPTIONAL) Print information during training
            
            if(print_var and epoch % step_show == 0):
                print("Epoch: {} ({:.2f}%) - Subject: {} - Repetition: {}".format(epoch, epoch/epochs * 100, idx, rep))
                
                print("\tLoss (TRAIN)\t: ", float(tmp_loss_train_total))
                print("\t\tReconstr (TRAIN)\t\t: ", float(tmp_loss_train_recon))
                print("\t\tKullback (TRAIN)\t\t: ", float(tmp_loss_train_kl))
                print("\t\tDiscriminator (TRAIN)\t: ", float(tmp_loss_train_discriminator), "\n")
                
                print("\tLoss (VAL)\t\t: ", float(tmp_loss_val_total))
                print("\t\tReconstr (VAL)\t\t\t: ", float(tmp_loss_val_recon))
                print("\t\tKullback (VAL)\t\t\t: ", float(tmp_loss_val_kl))
                print("\t\tDiscriminator (VAL)\t: ", float(tmp_loss_val_discriminator), "\n")
                
                print("\tLoss (TEST)\t\t: ", float(tmp_loss_test_total))
                print("\t\tReconstr (TEST)\t\t\t: ", float(tmp_loss_test_recon))
                print("\t\tKullback (TEST)\t\t\t: ", float(tmp_loss_test_kl))
                print("\t\tDiscriminator (TEST)\t: ", float(tmp_loss_test_discriminator))
                print("\t\tAccuracy (TEST)\t\t\t: ", float(tmp_accuracy_test), "\n")
                
                print("\tBest loss val\t\t: ", float(best_loss_validation))
                print("\tBest Accuracy test (1)\t: ", float(best_accuracy_test))
                print("\tBest Accuracy test (2)\t: ", float(np.max(accuracy_test)))
                print("\tNo Improvement\t\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break; 
                
            if(use_lr_scheduler): scheduler.step()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # (OPTIONAL) Save the model
    if(save_model):  
        save_path_END = "Saved model/eeg_framework"
        if(use_shifted_VAE_loss): save_path_END = save_path_END + "_advance_loss"
        else: save_path_END = save_path_END + "_normal_loss"
        save_path_END = save_path_END + "_" + str(rep)
        save_path_END = save_path_END + "_" + str(epoch) + ".pth"
        
        torch.save(eeg_framework.state_dict(), save_path_END)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Backup dictionary with saved results
    tmp_backup_dict = {}
    
    # Saved accuracy during training
    if(measure_accuracy): 
        accuracy_test_list.append(accuracy_test)
        tmp_backup_dict['accuracy_test_list'] = accuracy_test_list
    
    # Saving accuracy when min loss is reached for test set (during actual repetition)
    subject_accuracy_during_epochs_for_repetition_LOSS.append(np.asanyarray(subject_accuracy_during_epochs_LOSS).T)
    best_subject_accuracy_for_repetition_LOSS[:, rep] = subject_accuracy_during_epochs_LOSS[-1]
    tmp_backup_dict['best_subject_accuracy_for_repetition_LOSS'] = best_subject_accuracy_for_repetition_LOSS
    
    # Saving kappa score when min loss is reached for the test set (during actual repetition)
    subject_kappa_score_during_epochs_for_repetition_LOSS.append(np.asanyarray(subject_kappa_score_during_epochs_LOSS).T)
    best_subject_kappa_score_for_repetition_LOSS[:, rep] = subject_kappa_score_during_epochs_LOSS[-1]
    tmp_backup_dict['best_subject_kappa_score_for_repetition_LOSS'] = best_subject_kappa_score_for_repetition_LOSS
    
    # Saved the best accuracy reached on the test set during the training
    best_average_accuracy_for_repetition.append(np.max(accuracy_test))
    tmp_backup_dict['best_average_accuracy_for_repetition'] = best_average_accuracy_for_repetition

    # Saved the accuracy and the kappa score reached at the end of the training
    tmp_accuracy_end, tmp_kappa_score_end = measureSingleSubjectAccuracyAndKappaScore(eeg_framework, merge_list, dataset_type, normalize_trials, use_reparametrization_for_classification, device)
    subject_accuracy_test = np.asarray(tmp_accuracy_end)
    best_subject_accuracy_for_repetition_END[:, rep] = subject_accuracy_test
    tmp_backup_dict['best_subject_accuracy_for_repetition_END'] = best_subject_accuracy_for_repetition_END
    subject_kappa_score_test = np.asarray(tmp_kappa_score_end)
    best_subject_kappa_score_for_repetition_END[:, rep] = subject_kappa_score_test
    tmp_backup_dict['best_subject_kappa_score_for_repetition_END'] = best_subject_kappa_score_for_repetition_END
    
    parameter_train.append(tmp_parameter_train)
    tmp_backup_dict['parameter_train'] = parameter_train
    
    # Save the dictionary
    savemat("Saved Model/TMP RESULTS/backup.mat", tmp_backup_dict)
    
    # Clean the variable
    del full_dataset, train_dataset, train_dataloader, validation_dataset, validation_dataloader, test_dataset, test_dataloader

    

#%%

plt.figure()
plt.plot(total_loss_train)
plt.plot(total_loss_test)
plt.legend(["Train", "Test"])
plt.title("Total Loss")

plt.figure()
plt.plot(np.asarray(reconstruction_loss_train))
plt.plot(np.asarray(reconstruction_loss_test))
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

# visualizeHiddenSpace(eeg_framework.vae, train_dataset, idx_hidden_space = idx_hidden_space, sampling = False, n_elements = len(train_dataset), device = 'cuda')
# visualizeHiddenSpace(eeg_framework.vae, test_dataset, idx_hidden_space = idx_hidden_space, sampling = False, n_elements = len(test_dataset), device = 'cuda')

#%%