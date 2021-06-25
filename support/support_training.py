import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os

import torch
from torch import nn

#%%

def VAE_loss(x, x_out, mu, log_var, alpha = 1):
    # Kullback-Leibler Divergence
    kl_loss =  (-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim = 1)).mean(dim = 0)        
    
    # Reconstruction loss
    recon_loss_criterion = nn.MSELoss() #Reconstruction Loss
    recon_loss = recon_loss_criterion(x,x_out)
    
    # Total loss
    loss = recon_loss * alpha + kl_loss
    
    return loss


def advanceEpoch(vae, device, dataloader, loss_fn, optimizer = None, is_train = True, alpha = 1, print_var = False):
    if(is_train): vae.train()
    else: vae.eval()
    
    # Track variable
    i = 0
    tot_loss = 0
    
    for sample_data_batch, sample_label_batch in dataloader:
        if(is_train): # Train step (keep track of the gradient)
            x = sample_data_batch.to(device)
            vae.to(device)
            x_out, mu, log_var = vae(x)
            loss = loss_fn(x, x_out, mu, log_var, alpha)
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x = sample_data_batch.to(device)
                vae.to(device)
                x_out, mu, log_var = vae(x)
                loss = loss_fn(x, x_out, mu, log_var, alpha)
        
        # Backward/Optimization pass (only in training)
        if(is_train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        tot_loss += loss
            
        
        if(i % 3 == 0 and print_var): 
            print("     " + round(i/len(dataloader) * 100, 2), "%")
            print("     Actual loss: ", loss)
            print("     Total loss: ", tot_loss)
            
        i += 1
        
    return tot_loss

#%%

class Pytorch_Dataset_HGD(torch.utils.data.Dataset):
    
    # Inizialization method
    def __init__(self, path, n_elements = -1):
        tmp_list = []
        
        # Read all the file in the folder and return them as list of string
        for element in os.walk(path): tmp_list.append(element)
        
        # print(path, tmp_list)
        self.path = tmp_list[0][0]
        self.file_list = tmp_list[0][2]
        
        self.path_list = []
        
        for i in range(len(self.file_list)): 
            file = self.file_list[i]
            self.path_list.append(path + file)
            
            if(i >= (n_elements - 1) and n_elements != -1): break
            
        self.path_list = np.asarray(self.path_list)
        
        # Retrieve dimensions
        tmp_trial = loadmat(self.path_list[0])['trial']
        self.channel = tmp_trial.shape[0]
        self.samples = tmp_trial.shape[1]
            
        
    def __getitem__(self, idx, tensor = True):
        tmp_dict = loadmat(self.path_list[idx])
        
        # Retrieve and save trial 
        trial = tmp_dict['trial']
        trial = np.expand_dims(trial, axis = 0)
        
        # Retrieve label. Since the original label are in the range 1-4 I shift them in the range 0-3 for the NLLLoss() loss function
        label = int(tmp_dict['label']) - 1
                
        # Convert to PyTorch tensor
        if(tensor):
            trial = torch.from_numpy(trial).float()
            label = torch.tensor(label).long()
        
        return trial, label
    
    def __len__(self):
        return len(self.path_list)