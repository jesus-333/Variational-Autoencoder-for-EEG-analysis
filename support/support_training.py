import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from scipy.io import loadmat
from sklearn.metrics import cohen_kappa_score

import torch
from torch import nn

from support_datasets import PytorchDatasetEEGSingleSubject

#%% Loss functions


def VAE_loss(x, x_r, mu_q, log_var, alpha = 1, beta = 1, L2_loss_type = 0):
    """
    Standard loss of the VAE. 
    It return the reconstruction loss between x and x_r and the Kullback between a standard normal distribution and the ones defined by sigma and log_var
    It also return the sum of the two.

    """
    
    # Kullback-Leibler Divergence
    sigma_p = torch.ones(log_var.shape).to(log_var.device) # Standard deviation of the target standard distribution
    mu_p = torch.zeros(mu_q.shape).to(mu_q.device) # Mean of the target gaussian distribution
    sigma_q = torch.sqrt(torch.exp(log_var)) # standard deviation obtained from the VAE
    kl_loss = KL_Loss(sigma_p, mu_p, sigma_q, mu_q)
    # N.b. Due to implementation reasons I pass to the function the STANDARD DEVIATION, i.e. the NON-SQUARED VALUE
    # When the variance is needed inside the function the sigmas are eventually squared
    
    # Old KL Loss (Simplified version with sigma_p = 1 and mu_p = 0)
    # kl_loss =  (-0.5 * (1 + log_var - torch.exp(log_var) - mu**2).sum(dim = 1)).mean(dim = 0)
    
    # Reconstruction loss 
    # TODO return to normal recon loss
    if(L2_loss_type == 0):
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x_r, x)
    elif(L2_loss_type == 1): recon_loss = L2Loss_row_by_row(x, x_r, alpha) # P.s. The alpha is needed because it is applied to every row
    elif(L2_loss_type == 2): recon_loss = advance_recon_loss(x, x_r)
        
    
    # print(recon_loss.shape, "aaa")
    
    # Total loss
    if(L2_loss_type == 1): vae_loss = recon_loss + kl_loss * beta
    else: vae_loss = recon_loss * alpha + kl_loss * beta  
    
    return vae_loss, recon_loss * alpha, kl_loss * beta

def shifted_VAE_loss(x, x_r, mu_q, log_var, true_label, alpha = 1, beta = 1, shift_from_center = 0.5):
    """
    Modified VAE loss where each class is econded with a different distribution.
    In this case the Kullback for each class is different.
    """
    
    # Target distributions
    sigma_p = torch.ones(log_var.shape).to(log_var.device)
    mu_p = torch.zeros(mu_q.shape).to(mu_q.device)
    for i in range(4):mu_p[true_label == i, :] = constructMuTargetTensor(mu_p, shift_from_center, label = i)
    
    sigma_q = torch.sqrt(torch.exp(log_var)) # standard deviation obtained from the VAE
    kl_loss = KL_Loss(sigma_p, mu_p, sigma_q, mu_q)
 
    # Reconstruction loss
    recon_loss_criterion = nn.MSELoss()
    recon_loss = recon_loss_criterion(x_r, x)
    
    # Total loss
    vae_loss = recon_loss * alpha + kl_loss * beta
    
    return vae_loss, recon_loss, kl_loss * beta


def L2Loss_row_by_row(x, x_r, alpha = 1):
    """
    Modified version of L2 loss. Instead of simply appli the L2 loss at the entire matrix the methods applied it row by row.
    The input are the dataset data (x) and the reconstructed data (x_r)
    
    N.B. Remember that since x and x_r are created inside the advanceEpochV2 function and therefore they came from a dataloader.
         This means that they have 4 dimension: (batch_size, depth, channel, time samples).
    """
    
    # Reconstruction loss criterion
    recon_loss_criterion = nn.MSELoss()
    
    # Tensor to save the total loss
    recon_loss = torch.zeros(1).to(x.device)
    
    # Cycle through batch elements
    for i in range(x_r.shape[0]):
        # Extract i-th element of the batch. Also since depth dimension (see above) is 1 I simply "remove it" by taking the first element (index 0) of that dimension
        x_i = x[i, 0]
        x_r_i = x_r[i, 0]
        
        # Cycle through channels
        for ch in range(x_r_i.shape[0]):
            # Extract channel
            x_i_ch = x_i[ch].to()
            x_r_i_ch = x_r_i[ch]
            
            # Evaluate loss
            recon_loss += alpha * recon_loss_criterion(x_i_ch, x_r_i_ch)
    
    return recon_loss


def advance_recon_loss(x, x_r, std_r):
    """
    Advance versione of the recontruction loss for the VAE when the output distribution is gaussian.
    Instead of the simple L2 loss we use the log-likelihood formula so we can also encode the variance in the output of the decoder.
    Input parameters:
      x = Original data
      x_r = mean of the reconstructed output
      std_r = standard deviation of the reconstructed output. This is a scalar value.
    
    More info: 
    https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
    https://arxiv.org/pdf/2006.13202.pdf
    """
    
    total_loss = 0
    
    # MSE part
    mse_core = torch.pow((x - x_r), 2).sum(1)/ x.shape[1]
    mse_scale = (x[0].shape[0]/(2 * torch.pow(std_r, 2)))
    total_loss += (mse_core * mse_scale)
    
    # Variance part
    # total_loss += x[0].shape[0] * torch.log(std_r).mean()
    
    return total_loss
    

def constructMuTargetTensor(mu_p, shift_from_center, label):
    mu_length = mu_p.shape[1]
    target = torch.ones(mu_length).to(mu_p.device)
    
    if(label == 0): 
        target[0:int(mu_length/2)] = shift_from_center
        target[int(mu_length/2):] = 0
    if(label == 1): 
        target[0:int(mu_length/2)] = 0
        target[int(mu_length/2):] = shift_from_center
    if(label == 2): 
        target[0:int(mu_length/2)] = -shift_from_center
        target[int(mu_length/2):] = 0
    if(label == 3): 
        target[0:int(mu_length/2)] = 0
        target[int(mu_length/2):] = -shift_from_center
        
    return target
        

def KL_Loss(sigma_p, mu_p, sigma_q, mu_q):
    """
    General function for a KL loss with specified the paramters of two gaussian distributions p and q
    The parameter must be sigma (standard deviation) and mu (mean).
    The order of the parameter must be the following: sigma_p, mu_p, sigma_q, mu_q
    """
    
    tmp_el_1 = torch.log(sigma_q/sigma_p)
    
    tmp_el_2_num = torch.pow(sigma_q, 2) + torch.pow((mu_q - mu_p), 2)
    tmp_el_2_den = 2 * torch.pow(sigma_p, 2)
    tmp_el_2 = tmp_el_2_num / tmp_el_2_den
    
    kl_loss = - (tmp_el_1  - tmp_el_2 + 0.5)
    # print(kl_loss.shape)
    # print(kl_loss.sum(dim = 1).shape)
    
    return kl_loss.sum(dim = 1).mean(dim = 0)


def classifierLoss(predict_label, true_label):
    classifier_loss_criterion = torch.nn.NLLLoss()
    # classifier_loss_criterion = torch.nn.CrossEntropyLoss()
    
    return classifier_loss_criterion(predict_label, true_label)


def VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_shifted_VAE_loss = False, alpha = 1, beta = 1, gamma = 1):
    # VAE loss (reconstruction + kullback)
    if(use_shifted_VAE_loss):
        shift_from_center = 0.7
        vae_loss, recon_loss, kl_loss = shifted_VAE_loss(x, x_r, mu, log_var, true_label, alpha, beta, shift_from_center)
    else:
        vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha, beta)
    
    # Classifier (discriminator) loss
    classifier_loss = classifierLoss(predict_label, true_label) 
    
    # Total loss
    total_loss = vae_loss + classifier_loss * gamma
    
    # print("Print Loss:")
    # print("Total:", total_loss)
    # print("Class:", classifier_loss)
    # print("Recon:", recon_loss)
    # print("kl:   ", kl_loss)
    
    return total_loss, recon_loss, kl_loss, classifier_loss * gamma

#%% Training functions

def advanceEpochV1(vae, device, dataloader, optimizer, is_train = True, alpha = 1, print_var = False):
    """
    Function used to advance one epoch of training in training script 1 (Only VAE)
    
    """
    
    if(is_train): vae.train()
    else: vae.eval()
    
    # Track variable
    i = 0
    tot_vae_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    
    for sample_data_batch, sample_label_batch in dataloader:
        # Move data and vae to device
        x = sample_data_batch.to(device)
        vae.to(device)
        
        if(is_train): # Train step (keep track of the gradient)
            # Zeros past gradients
            optimizer.zero_grad()
            
            # VAE works
            x_r, mu, log_var = vae(x)
            
            # Evaluate loss
            vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
            
            # Backward/Optimization pass
            vae_loss.backward()
            optimizer.step()
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x_r, mu, log_var = vae(x)
                vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
            
        # Compute the total loss
        tot_vae_loss += vae_loss
        tot_recon_loss += recon_loss
        tot_kl_loss += kl_loss
            
        
        if(i % 3 == 0 and print_var): 
            print("     " + round(i/len(dataloader) * 100, 2), "%")
            print("     Actual loss: ", vae_loss)
            print("     Total loss: ", tot_vae_loss)
            
        i += 1
        
    return tot_vae_loss, tot_recon_loss, tot_kl_loss


def advanceEpochV2(eeg_framework, device, dataloader, optimizer = None, is_train = True, use_shifted_VAE_loss = False, alpha = 1, beta = 1, gamma = 1, print_var = False):
    """
    Function used to advance one epoch of training in training scripts 2 and 3 (VAE + Classifier trained jointly)
    
    """
    
    if(is_train): eeg_framework.train()
    else: eeg_framework.eval()
    
    # Track variable
    i = 0
    tot_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    tot_discriminator_loss = 0
    
    for sample_data_batch, sample_label_batch in dataloader:
        # Move data, label and netowrks to device
        x = sample_data_batch.to(device)
        true_label = sample_label_batch.to(device)
        eeg_framework.to(device)
        
        if(is_train): # Train step (keep track of the gradient)
            # Zeros past gradients
            optimizer.zero_grad()
            
            # Networks forward pass
            x_r, mu, log_var, predict_label = eeg_framework(x)
            
            # Loss evaluation
            total_loss, recon_loss, kl_loss, discriminator_loss = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_shifted_VAE_loss, alpha, beta, gamma)
            
            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()    
        
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x = sample_data_batch.to(device)
                eeg_framework.to(device)
                x_r, mu, log_var, predict_label = eeg_framework(x)
                total_loss, recon_loss, kl_loss, discriminator_loss = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_shifted_VAE_loss, alpha, beta, gamma)
               
        # Compute the total loss
        tot_loss += total_loss
        tot_recon_loss += recon_loss
        tot_kl_loss += kl_loss
        tot_discriminator_loss += discriminator_loss
            
        
        if(i % 3 == 0 and print_var): 
            print("     " + round(i/len(dataloader) * 100, 2), "%")
            print("     Actual loss: ", total_loss)
            print("     Total loss: ", tot_loss)
            
        i += 1
        
    return tot_loss, tot_recon_loss, tot_kl_loss, tot_discriminator_loss


def advanceEpochV3(model, model_type, device, dataloader, optimizer = None, is_train = True, alpha = 1, beta = 1):
    """
    Function used to advance one epoch of training in training scripts 4 (VAE + Classifier trained separately)

    Parameters
    ----------
    model : PyTorch model
        Model to train. Can be the VAE or the classifier.
    model_type : int
        Specify if the model is the VAE or the classifier. 0 for VAE and 1 for classifier

    """
    
    # Check the model type
    if(model_type != 0 and model_type != 1): raise Exception("\"model_type\" must have value 0 (VAE) or 1 (classifier)")
    
    # Check if is trained or test epoch
    if(is_train): model.train()
    else: model.eval()
    
    # Track variable
    i = 0
    tot_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    tot_discriminator_loss = 0
    
    for sample_data_batch, sample_label_batch in dataloader:
        # Move data, label and netowrk to device
        x = sample_data_batch.to(device)
        model.to(device)
        
        if(is_train): # Train step (keep track of the gradient)
            # Zeros past gradients
            optimizer.zero_grad()
            
            # Networks forward pass and loss evaluation            
            if(model_type == 0): # VAE
                x_r, mu, log_var = model(x)
                total_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
            elif (model_type == 1): # Classifier
                mu, log_var = model[0].encoder(x)
                z = torch.cat((mu, log_var), dim = 1)
                predict_label = model[1].classifier(z)
                true_label = sample_label_batch.to(device)
                total_loss = classifierLoss(predict_label, true_label)

            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()    
        
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x = sample_data_batch.to(device)
                model.to(device)
                
                if(model_type == 0): # VAE
                    x_r, mu, log_var = model(x)
                    total_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
                elif (model_type == 1): # Classifier
                    mu, log_var = model[0].encoder(x)
                    z = torch.cat((mu, log_var), dim = 1)
                    predict_label = model[1].classifier(z)
                    true_label = sample_label_batch.to(device)
                    total_loss = classifierLoss(predict_label, true_label)
        
        # Compute the total loss
        if(model_type == 0):
            tot_loss += total_loss
            tot_recon_loss += recon_loss
            tot_kl_loss += kl_loss
        elif(model_type == 1):
            tot_discriminator_loss += total_loss
            
    if(model_type == 0):
        return tot_loss, tot_recon_loss, tot_kl_loss
    elif(model_type == 1):
        return tot_discriminator_loss 
        

#%%

def measureAccuracyAndKappaScore(vae, classifier, dataset, device = 'cpu', use_reparametrization_for_classification = False, print_var = False):
    """
    Functions used to measure accuracy in training script 2 and 3.

    """
    
    # Move classifier and vae to the device
    vae = vae.to(device)
    classifier = classifier.to(device)
    
    # Set the vae and the classifier in evaluation mode
    vae.eval()
    classifier.eval()
    
    # Tracking varaible to count the number of correct prediction
    correct_classification = 0
    
    # List of labels
    true_labels_list = []
    predict_labels_list = []
    
    # Iterate through element of the dataset
    for i in range(len(dataset)):
        if(print_var): print("Completition: {}".format(round(i/len(dataset) * 100, 2)))
        
        # Retrieve EEG data and move to device
        x_eeg = dataset[i][0].unsqueeze(0)
        x_eeg = x_eeg.to(device)
        
        # Retrieve correct label
        true_label = dataset[i][1]
        
        # Forward pass through vae encoder
        mu, log_var = vae.encoder(x_eeg)
        
        # (OPTIONAL) Use reparametrization. Otherwise use directly the values of mu and log_var as inputs
        if(use_reparametrization_for_classification): z = vae.reparametrize(mu, log_var)
        else: z = torch.cat((mu, log_var), dim = 1)
        
        # Forward pass through classifier
        classifier_output = classifier(z)
        
        # Retrieve probability of each class and select the class with the highest probability
        predict_prob = np.squeeze(torch.exp(classifier_output).cpu().detach().numpy())
        predict_label = np.argmax(predict_prob)

        # Increase the counter of 1 if the prediction is correct        
        if(predict_label == true_label): correct_classification += 1
        
        # Save labels
        true_labels_list.append(float(true_label))
        predict_labels_list.append(float(predict_label))
    
    # Evaluate percentage of correct prediction
    accuracy = correct_classification / len(dataset)
    
    # Evaluate Cohen's Kappa Score
    kappa_score = cohen_kappa_score(true_labels_list, predict_labels_list)
    
    return accuracy, kappa_score


def measureSingleSubjectAccuracyAndKappaScore(eeg_framework, merge_list, dataset_type, normalize_trials = False, use_reparametrization_for_classification = False, device = torch.device("cpu")):
    """
    Support function used when the framework is trained with the merge dataset. Used in training script 2 and 3.
    It measures the accuracy for each subject and return a list of accuracies.

    """
    
    # List to save accuracy and kappa score
    accuracy_list = []
    kappa_score_list = []
    
    # Cycle thorugh subjects
    for idx in merge_list: 
        # Dataset creation
        if(dataset_type == 'HGD'):
            path = 'Dataset/HGD/Train/{}/'.format(idx)
        elif(dataset_type == 'D2A'):
            path = 'Dataset/D2A/v2_raw_128/Train/{}/'.format(idx)
        dataset_subject = PytorchDatasetEEGSingleSubject(path, normalize_trials = normalize_trials)
        
        # Accuracy and kappa score evaluation
        subject_accuracy, subject_kappa_score = measureAccuracyAndKappaScore(eeg_framework.vae, eeg_framework.classifier, dataset_subject, device, use_reparametrization_for_classification)
        
        # Save results
        accuracy_list.append(subject_accuracy)
        kappa_score_list.append(subject_kappa_score)
        
    return accuracy_list, kappa_score_list
