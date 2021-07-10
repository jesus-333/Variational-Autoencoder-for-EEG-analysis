import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys

import torch
from torch import nn

#%% Loss functions

def VAE_loss(x, x_r, mu, log_var, alpha = 1):
    # Kullback-Leibler Divergence
    sigma_p = torch.ones(log_var.size()).to(log_var.device)
    mu_p = torch.zeros(mu.size()).to(mu.device)
    kl_loss = KL_Loss(sigma_p, mu_p, torch.sqrt(torch.exp(log_var)), mu)
    
    # Old KL Loss (Simplified version with sigma_p = 1 and mu_p = 0)
    # kl_loss =  (-0.5 * (1 + log_var - torch.exp(log_var) - mu**2).sum(dim = 1)).mean(dim = 0)
    
    # Reconstruction loss
    recon_loss_criterion = nn.MSELoss()
    # recon_loss_criterion = nn.BCELoss()
    recon_loss = recon_loss_criterion(x_r, x)
    
    # Total loss
    vae_loss = recon_loss * alpha + kl_loss
    
    return vae_loss, recon_loss, kl_loss


def advance_VAE_loss(x, x_r, mu, log_var, true_label, alpha = 1, shift_from_center = 0.5):
    """
    Modified VAE loss where each class is econded with a different distribution
    """
    
    # Target distributions
    sigma_p = torch.ones(log_var.size()).to(log_var.device)
    mu_p = torch.zeros(mu.size()).to(mu.device)
    mu_p[true_label == 0, 0:2] = torch.tensor([shift_from_center, 0]).to(mu_p.device)
    mu_p[true_label == 1, 0:2] = torch.tensor([0, shift_from_center]).to(mu_p.device)
    mu_p[true_label == 2, 0:2] = torch.tensor([-shift_from_center, 0]).to(mu_p.device)
    mu_p[true_label == 3, 0:2] = torch.tensor([0, -shift_from_center]).to(mu_p.device)
    
    kl_loss = KL_Loss(sigma_p, mu_p, torch.sqrt(torch.exp(log_var)), mu)
    
    # Reconstruction loss
    recon_loss_criterion = nn.MSELoss()
    # recon_loss_criterion = nn.BCELoss()
    recon_loss = recon_loss_criterion(x_r, x)
    
    # Total loss
    vae_loss = recon_loss * alpha + kl_loss
    
    return vae_loss, recon_loss, kl_loss
    
    

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
    
    return kl_loss.sum(dim = 1).mean(dim = 0)

def VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_advance_vae_loss = False, alpha = 1):
    # VAE loss (reconstruction + kullback)
    if(use_advance_vae_loss):
        shift_from_center = 0.7
        vae_loss, recon_loss, kl_loss = advance_VAE_loss(x, x_r, mu, log_var, true_label, alpha, shift_from_center)
    else:
        vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
    
    # Discriminator loss
    discriminator_loss_criterion = torch.nn.NLLLoss()
    # discriminator_loss_criterion = torch.nn.CrossEntropyLoss()
    discriminator_loss = discriminator_loss_criterion(predict_label, true_label)
    
    # Total loss
    total_loss = vae_loss + discriminator_loss
    
    return total_loss, recon_loss, kl_loss, discriminator_loss

#%% Training functions

def advanceEpochV1(vae, device, dataloader, optimizer = None, is_train = True, alpha = 1, print_var = False):
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
            
        # Save total loss
        tot_vae_loss += vae_loss
        tot_recon_loss += recon_loss
        tot_kl_loss += kl_loss
            
        
        if(i % 3 == 0 and print_var): 
            print("     " + round(i/len(dataloader) * 100, 2), "%")
            print("     Actual loss: ", vae_loss)
            print("     Total loss: ", tot_vae_loss)
            
        i += 1
        
    return tot_vae_loss, tot_recon_loss, tot_kl_loss


def advanceEpochV2(eeg_framework, device, dataloader, optimizer = None, is_train = True, use_advance_vae_loss = False, alpha = 1, print_var = False):
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
            
            # Networks works
            x_r, mu, log_var, predict_label = eeg_framework(x)
            
            # Loss evaluation
            total_loss, recon_loss, kl_loss, discriminator_loss = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_advance_vae_loss, alpha)
            
            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()    
        
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x = sample_data_batch.to(device)
                eeg_framework.to(device)
                x_r, mu, log_var, predict_label = eeg_framework(x)
                total_loss, recon_loss, kl_loss, discriminator_loss = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, use_advance_vae_loss, alpha)
        
            
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

#%%

def measureAccuracy(vae, classifier, dataset, device = 'cpu', print_var = False):
    # Set the classifier in evaluation mode
    classifier.eval()
    
    vae = vae.to(device)
    classifier = classifier.to(device)
    
    correct_classification = 0
    
    # Iterate through element of the dataset
    for i in range(len(dataset)):
        if(print_var): print("Completition: {}".format(round(i/len(dataset) * 100, 2)))
        x_eeg = dataset[i][0].unsqueeze(0)
        true_label = dataset[i][1]
        x_eeg = x_eeg.to(device)
        
        mu, log_var = vae.encoder(x_eeg)
        z = torch.cat((mu, log_var), dim = 1)
        classifier_output = classifier(z)
        
        predict_prob = np.squeeze(torch.exp(classifier_output).cpu().detach().numpy())
        predict_label = np.argmax(predict_prob)
        
        if(predict_label == true_label): correct_classification += 1
        
    accuracy = correct_classification / len(dataset)
    
    return accuracy