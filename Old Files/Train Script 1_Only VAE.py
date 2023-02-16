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
from support.support_training import VAE_loss, advanceEpochV1, Pytorch_Dataset_HGD

#%% Settings

hidden_space_dimension = 256

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
            tmp_loss_train_total, tmp_loss_train_recon, tmp_loss_train_kl = advanceEpochV1(vae, device, train_dataloader, optimizer, is_train = True, alpha = alpha)
            total_loss_train.append(float(tmp_loss_train_total))
            
            # Testing phase
            tmp_loss_test_total, tmp_loss_test_recon, tmp_loss_test_kl = advanceEpochV1(vae, device, test_dataloader, is_train = False, alpha = alpha)
            total_loss_test.append(float(tmp_loss_test_total))
            
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
                print("\t\tReconstr (TRAIN)\t: ", float(tmp_loss_train_recon))
                print("\t\tKullback (TRAIN)\t: ", float(tmp_loss_train_kl), "\n")
                
                print("\tLoss (TEST)\t\t: ", float(tmp_loss_test_total))
                print("\t\tReconstr (TEST)\t: ", float(tmp_loss_test_recon))
                print("\t\tKullback (TEST)\t: ", float(tmp_loss_test_kl), "\n")
                
                print("\tBest loss test\t: ", float(best_loss_test))
                print("\tNo Improvement\t: ", int(epoch_with_no_improvent))
                
                print("- - - - - - - - - - - - - - - - - - - - - - - - ")
                
            if(epoch_with_no_improvent > 50 and early_stop): 
                if(print_var): print("     JUMP\n\n")
                break;

#%% Test (Temporary)
idx_ch = np.random.randint(0, 127)

vae.cpu()
x = train_dataset[np.random.randint(0, 158)][0]
x_r = vae(x.unsqueeze(0))[0].squeeze()

# print("MSE LOSS (torch):\t\t", float(torch.nn.MSELoss()(x.squeeze(), x_r)))

x = x.squeeze().numpy()
x_r = x_r.detach().squeeze().numpy()

plt.plot(x[idx_ch])
plt.plot(x_r[idx_ch])
# plt.xlim([0, 1000])

# tot_mse = 0
# for ch in range(x.shape[0]):
#     for i in range(x.shape[1]):
#         tot_mse += (x[ch][i] - x_r[ch][i]) ** 2
        
# print("MSE loss (handamade):\t", tot_mse / (x.shape[0] * x.shape[1]))

#%%

vae.cuda()
# ax = plt.axes(projection='3d')

mu_lists = {0:[], 1:[], 2:[], 3:[]}
std_lists = {0:[], 1:[], 2:[], 3:[]}

n_elements = 800
for i in range(n_elements):
    print("Completition: {}".format(round(i/n_elements, 2) * 100))
    
    x_eeg = train_dataset[i][0].unsqueeze(0).cuda()
    label = int(train_dataset[i][1])
    
    z = vae.encoder(x_eeg)
    
    mu = z[0].cpu().squeeze().detach().numpy()
    
    # N.B. Since I obtain the logarimt of the variance from the VAE I moltiply for 0.5 = 1/2 to obtain the standard deviation
    std = torch.exp(0.5 * z[1]).cpu().squeeze().detach().numpy()
    
    x = np.random.normal(mu[0], std[0], 1)
    y = np.random.normal(mu[1], std[1], 1)
    # z = np.random.normal(mu[2], var[2], 1)
    
    if(label == 0): 
        plt.plot(x, y, 'ko')
    elif(label == 1): 
        plt.plot(x, y, 'ro')
    elif(label == 2): 
        plt.plot(x, y, 'yo')
    elif(label == 3): 
        plt.plot(x, y, 'bo')
        
    mu_lists[label].append(mu)
    std_lists[label].append(std)
    
    # if(label == 0): ax.scatter3D(x, y, z, c ='g', marker = 'o')
    # elif(label == 1): ax.scatter3D(x, y, z, c ='r', marker = '+')
    # elif(label == 2): ax.scatter3D(x, y, z, c ='b', marker = '^')
    # elif(label == 3): ax.scatter3D(x, y, z, c ='k', marker = 'v')
    