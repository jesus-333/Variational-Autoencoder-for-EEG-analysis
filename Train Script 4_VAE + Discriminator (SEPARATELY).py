"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the VAE and Discriminator SEPARATELY for INTRA subject classification.
The train datasets are merged toghether and then the network is tested on the test datasets separately.

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

from support.VAE_EEGNet import EEGNetVAE, CLF_V1
from support.support_training import advanceEpochV3, measureAccuracy, measureSingleSubjectAccuracy
from support.support_datasets import PytorchDatasetEEGSingleSubject, PytorchDatasetEEGMergeSubject
from support.support_visualization import visualizeHiddenSpace

#%% Settings

dataset_type = 'D2A'

hidden_space_dimension = 64
# hidden_space_dimension = 32 # TODO Remove

print_var = True
tracking_input_dimension = True

epochs = 500
batch_size = 15
learning_rate = 1e-3
alpha = 0.01 #TODO Tune alpha
repetition = 7

normalize_trials = False
use_reparametrization = False 
merge_subject = True
execute_test_epoch = True
early_stop = False
use_advance_vae_loss = False
measure_accuracy = True
save_model = True

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

if(merge_subject): 
    merge_list = idx_list.copy()
    idx_list = [1]
    

for rep in range(repetition):
    for idx in idx_list:
        
        subject_accuracy_during_epochs_LOSS = []
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Train dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Train/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Train/{}/'.format(idx)
            
        if(merge_subject):
            train_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = merge_list)
        else:
            train_dataset = PytorchDatasetEEGSingleSubject(path, normalize_trials = normalize_trials)
        
        # Train dataloader
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TRAIN dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Test dataset
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Test/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Test/{}/'.format(idx)
        
        if(merge_subject):
            test_dataset = PytorchDatasetEEGMergeSubject(path[0:-2], idx_list = merge_list)
        else:
            test_dataset = PytorchDatasetEEGSingleSubject(path, normalize_trials = normalize_trials)
          
        # Test dataloader
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
        if(print_var): print("TEST dataset and dataloader created\n")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # VAE creation
        C = train_dataset[0][0].shape[1]
        T = train_dataset[0][0].shape[2]
        vae = EEGNetVAE(C = C, T = T, hidden_space_dimension = hidden_space_dimension, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        if(print_var): print("VAE created")
        
        # VAE Optimizer
        optimizer_vae = torch.optim.AdamW(vae.parameters(), lr = learning_rate, weight_decay = 1e-5)
        
        # Loss tracking variables VAE (TRAIN)
        total_loss_train = []
        reconstruction_loss_train = []
        kl_loss_train = []

        # Loss tracking variables VAE (TEST)
        total_loss_test = []
        reconstruction_loss_test = []
        kl_loss_test = []
        
        best_loss_test = sys.maxsize
        best_accuracy_test = sys.maxsize
        epoch_with_no_improvent = 0
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # VAE Training
        
        for epoch in range(epochs):
            
            # Training phase
            tmp_loss_train = advanceEpochV3(vae, 0, device, train_dataloader, optimizer_vae, is_train = True, alpha = alpha)
            tmp_loss_train_total = tmp_loss_train[0]
            tmp_loss_train_recon = tmp_loss_train[1]
            tmp_loss_train_kl = tmp_loss_train[2]
            
            # Save train losses
            total_loss_train.append(float(tmp_loss_train_total))
            reconstruction_loss_train.append(float(tmp_loss_train_recon))
            kl_loss_train.append(float(tmp_loss_train_kl))
            
            # Testing phase
            if(execute_test_epoch): tmp_loss_test = advanceEpochV3(vae, 0, device, test_dataloader, is_train = False, alpha = alpha)
            else: tmp_loss_test = [sys.maxsize, sys.maxsize, sys.maxsize]
            tmp_loss_test_total = tmp_loss_test[0]
            tmp_loss_test_recon = tmp_loss_test[1] 
            tmp_loss_test_kl = tmp_loss_test[2]
            
            # Save tet losses
            total_loss_test.append(float(tmp_loss_test_total))
            reconstruction_loss_test.append(float(tmp_loss_test_recon))
            kl_loss_test.append(float(tmp_loss_test_kl))
            
            

            
            if(tmp_loss_test_total < best_loss_test):
                # Update loss and accuracy (N.b. the best accuracy is intended as the accuracy when there is a new min loss)
                best_loss_test = tmp_loss_test_total
                
                # Reset counter
                epoch_with_no_improvent = 0
               
                # TODO (OPTIONAL) Save the model
                # if(save_model):  
                #     save_path = "Saved model/eeg_framework"
                #     if(use_advance_vae_loss): save_path = save_path + "_advance_loss"
                #     else: save_path = save_path + "_normal_loss"
                #     save_path = save_path + "_" + str(rep)
                #     save_path = save_path + "_" + str(epoch) + ".pth"
                    
                #     torch.save(eeg_framework.state_dict(), save_path)
                    
                #     if(print_var): print("SAVED MODEL AT EPOCH: ", epoch)
   
            else: 
                epoch_with_no_improvent += 1
                
                
            
            if(print_var and epoch % step_show == 0):
                print("Epoch: {} ({:.2f}%) - Subject: {} - Repetition: {}".format(epoch, epoch/epochs * 100, idx, rep))
                
                print("\tLoss (TRAIN)\t: ", float(tmp_loss_train_total))
                print("\t\tReconstr (TRAIN)\t\t: ", float(tmp_loss_train_recon))
                print("\t\tKullback (TRAIN)\t\t: ", float(tmp_loss_train_kl))
                
                print("\tLoss (TEST)\t\t: ", float(tmp_loss_test_total))
                print("\t\tReconstr (TEST)\t\t\t: ", float(tmp_loss_test_recon))
                print("\t\tKullback (TEST)\t\t\t: ", float(tmp_loss_test_kl))
                
                print("\tBest loss test\t\t: ", float(best_loss_test))
                print("\tNo Improvement\t\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break; 
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Freeze all VAE parameters
        for param in vae.parameters(): param.requires_grad = False
        
        # Classifier creation
        classifier = CLF_V1(hidden_space_dimension * 2)
        eeg_framework = torch.nn.Sequential(
            vae,
            classifier
        )
        
        # Classifier optimizer
        optimizer_clf = torch.optim.AdamW(classifier.parameters(), lr = learning_rate, weight_decay = 1e-5)
        
        # Loss tracking variables c\lassifier (TRAIN)
        discriminator_loss_train = []
        accuracy_train = []
        
        # Loss tracking variables classifier (TEST)
        discriminator_loss_test = []
        accuracy_test = []
        
        best_loss_test = sys.maxsize
        best_accuracy_test = sys.maxsize
        epoch_with_no_improvent = 0
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Classifier Training
        for epoch in range(epochs):
            # Training phase
            tmp_loss_train_discriminator = advanceEpochV3(eeg_framework, 1, device, train_dataloader, optimizer_clf, is_train = True)
            discriminator_loss_train.append(float(tmp_loss_train_discriminator))
            
            # Testing phase
            if(execute_test_epoch): tmp_loss_test_discriminator = advanceEpochV3(eeg_framework, 1, device, test_dataloader, is_train = False)
            else: tmp_loss_test_discriminator = sys.maxsize
            discriminator_loss_test.append(float(tmp_loss_test_discriminator))
            
            # Measure accuracy (OPTIONAL)
            if(measure_accuracy):
                eeg_framework.eval()
                tmp_accuracy_test = measureAccuracy(vae, classifier, test_dataset, device, use_reparametrization)
                accuracy_test.append(tmp_accuracy_test)
                eeg_framework.train()
                
            if(tmp_loss_test_total < best_loss_test):
                best_accuracy_test = tmp_accuracy_test
                
                if(measure_accuracy):
                    tmp_subject_accuracy = measureSingleSubjectAccuracy(eeg_framework, merge_list, dataset_type, use_reparametrization, device = device)
                    subject_accuracy_during_epochs_LOSS.append(tmp_subject_accuracy)
                    
                if(print_var and epoch % step_show == 0):
                    print("Epoch: {} ({:.2f}%) - Subject: {} - Repetition: {}".format(epoch, epoch/epochs * 100, idx, rep))
                    
                    print("\t\tDiscriminator (TRAIN)\t: ", float(tmp_loss_train_discriminator), "\n")
                    print("\t\tDiscriminator (TEST)\t: ", float(tmp_loss_test_discriminator))
                   
                    print("\t\tAccuracy (TEST)\t\t\t: ", float(tmp_accuracy_test), "\n")
                    
                    print("\tBest Accuracy test (1)\t: ", float(best_accuracy_test))
                    print("\tBest Accuracy test (2)\t: ", float(np.max(accuracy_test)))
                    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TODO (OPTIONAL) Save the model
    # if(save_model):  
    #     save_path_END = "Saved model/eeg_framework"
    #     if(use_advance_vae_loss): save_path_END = save_path_END + "_advance_loss"
    #     else: save_path_END = save_path_END + "_normal_loss"
    #     save_path_END = save_path_END + "_" + str(epoch) + ".pth"
        
    #     torch.save(vae.state_dict(), save_path_END)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TODO Backup dictionary with saved results
    # tmp_backup_dict = {}
    
    # # Saved accuracy during training
    # if(measure_accuracy): 
    #     accuracy_test_list.append(accuracy_test)
    #     tmp_backup_dict['accuracy_test_list'] = accuracy_test_list
    
    # # Saving accuracy when min loss is reached for test set (during actual repetition)
    # subject_accuracy_during_epochs_for_repetition_LOSS.append(np.asanyarray(subject_accuracy_during_epochs_LOSS).T)
    # best_subject_accuracy_for_repetition_LOSS[:, rep] = subject_accuracy_during_epochs_LOSS[-1]
    # tmp_backup_dict['best_subject_accuracy_for_repetition_LOSS'] = best_subject_accuracy_for_repetition_LOSS
    
    # # Saved the best accuracy reached on the test set during the training
    # best_average_accuracy_for_repetition.append(np.max(accuracy_test))
    # tmp_backup_dict['best_average_accuracy_for_repetition'] = best_average_accuracy_for_repetition

    # # Saved the accuracy reached at the end of the training
    # subject_accuracy_test = np.asarray(measureSingleSubjectAccuracy(eeg_framework, merge_list, dataset_type, normalize_trials, use_reparametrization, device))
    # best_subject_accuracy_for_repetition_END[:, rep] = subject_accuracy_test
    # tmp_backup_dict['best_subject_accuracy_for_repetition_END'] = best_subject_accuracy_for_repetition_END
    
    # # Save the dictionary
    # savemat("Saved Model/TMP RESULTS/backup.mat", tmp_backup_dict)

    

#%%

plt.figure()
plt.plot(total_loss_train)
plt.plot(total_loss_test)
plt.legend(["Train", "Test"])
plt.title("Total Loss")

plt.figure()
plt.plot(np.asarray(reconstruction_loss_train) * alpha)
plt.plot(np.asarray(reconstruction_loss_test) * alpha)
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