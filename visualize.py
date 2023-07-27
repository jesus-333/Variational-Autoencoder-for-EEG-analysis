# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:06:56 2023

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""
#%%

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
import library.training.train_generic as train_generic

#%% Load data

subj = [5]

dataset_config = cd.get_moabb_dataset_config(subj)
device = 'cpu'

C = 22
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
type_decoder = 0
parameters_map_type = 0
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)

train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Create model
model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
model = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
model.to(device)

#%% Test RECONSTRUCTION

tmp_dataset = test_dataset

epoch = 'BEST'
# path_weight = 'TMP_Folder/model_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_3_100_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_1_30_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_2_20_{}.pth'.format(epoch)

# path_weight = 'TMP_Folder/model_BEST_2.pth'
path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_3_not_normalized/model_BEST.pth'
# path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_5_not_normalized/model_BEST.pth'

path_weight = 'Saved Model/full_eeg/{}/model_BEST.pth'.format(subj[0])


model.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))

# subj 3 idx --> label
# 12 --> 0 (left), 33 --> 1 (right), 9 --> 2 (foot) 254 --> 3 (tongue)

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
label_to_ch = {'left' : 11, 'right' : 7, 'foot' : 9, 'tongue' : -1 }

with torch.no_grad():
    for i in range(200):
        # idx_trial = 178
        # idx_ch = 0
        # idx_trial = 214
        # idx_ch = 0
        idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        idx_ch =  int(np.random.randint(0, 22, 1))
        # x = tmp_dataset[idx_trial][0]

        # idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        x = tmp_dataset[idx_trial][0]
        label = label_dict[int(tmp_dataset[idx_trial][1])]
        if label == 'tongue': pass
        # idx_ch = label_to_ch[label]
        
        output = model(x.unsqueeze(0))
        x_r = output[0]

        t = np.linspace(2, 7, x.shape[-1])
        idx_t = np.logical_and(t > 2, t < 4)
        # idx_t = np.ones(len(t)) == 1

        x_plot = x.squeeze()[idx_ch, idx_t]
        x_r_plot = x_r.squeeze()[idx_ch, idx_t]

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t[idx_t], x_plot, label = 'Original Signal')
        plt.plot(t[idx_t], x_r_plot, label = 'Reconstructed Signal')
        plt.xlabel("Time [s]")
        plt.title("Subj {} - Trial {} - Label: {} - Ch: {}".format(subj[0], idx_trial, label, tmp_dataset.ch_list[idx_ch]))
        plt.legend()
        # plt.ylim([-40, 40])
        plt.grid(True)
        plt.show()

        # f, x_psd = signal.welch(x_plot, fs = 250,)
        # f, x_r_psd = signal.welch(x_r_plot * 20, fs = 250,)
        # plt.rcParams.update({'font.size': 20})
        # plt.figure(figsize = (15, 10))
        # plt.plot(f, x_psd)
        # plt.plot(f, x_r_psd)
        # plt.xlabel("Frequency [Hz]")
        # plt.title("Ampiezza originale")
        # plt.show()

#%% Test GENERATION different subject

epoch = 'BEST'
tmp_dataset = test_dataset

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
label_to_ch = {'left' : 11, 'right' : 7, 'foot' : 9, 'tongue' : -1 }

path_weight_1 = 'Saved model/hvae_dwt/subj_1_30_{}.pth'.format(epoch)
path_weight_2 = 'Saved model/hvae_dwt/subj_3_100_{}.pth'.format(epoch)

with torch.no_grad():
    for i in range(33):
        idx_ch =  int(np.random.randint(0, 22, 1))
        z = torch.randn(model.h_vae.hidden_space_shape)

        model.load_state_dict(torch.load(path_weight_1, map_location=torch.device('cpu')))
        x_1 = model.generate()

        model.load_state_dict(torch.load(path_weight_2, map_location=torch.device('cpu')))
        x_2 = model.generate()

        t = np.linspace(2, 6, x_1.shape[-1])

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t, x_1.squeeze()[idx_ch], label = 'Subject 1')
        plt.plot(t, x_2.squeeze()[idx_ch], label = 'Subject 3')
        plt.xlabel("Time [s]")
        plt.title("Ch: {}".format(tmp_dataset.ch_list[idx_ch]))
        plt.legend()
        plt.grid(True)
        plt.show()

#%% Compute DTW distance
subj = [3]
epoch = 'BEST'
dataset_config = cd.get_moabb_dataset_config(subj)

path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_3_not_normalized/model_BEST.pth'
# path_weight = 'TMP_Folder/full_eeg/{}/model_{}.pth'.format(subj[0], epoch)

model.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))

# dtw_distance_matrix_train = model.dtw_comparison(train_dataset[0:20][0], distance_function = 2)
# dtw_distance_matrix_train_2 = model.dtw_comparison_2(train_dataset[0:20][0], 'cuda')
# dtw_distance_matrix_test = model.dtw_comparison(test_dataset[:][0])

from fastdtw import fastdtw
# with torch.no_grad():
#     # Best dtw cuda subj 3
#     idx_trial = 228
#     idx_ch = 19
    
#     ouput = model(train_dataset[idx_trial][0].unsqueeze(0))
#     x = train_dataset[idx_trial][0][0, idx_ch]
#     x_r = ouput[0][0,0,idx_ch]
    
#     distance, _ = fastdtw(x, x_r, radius = 10)
#     print(distance)
    
    
#     # Worst dtw cuda subj 3
#     idx_trial = 214
#     idx_ch = 0
    
#     ouput = model(train_dataset[idx_trial][0].unsqueeze(0))
#     x = train_dataset[idx_trial][0][0, idx_ch]
#     x_r = ouput[0][0,0,idx_ch]
    
#     distance, _ = fastdtw(x, x_r, radius = 10)
#     print(distance)

def plot_quick(x, x_r):
    plt.figure(figsize = (15, 10))
    plt.plot(x, label = 'Original')
    plt.plot(x_r, label = 'Reconstrcuted')
    plt.legend()
    plt.grid(True)
    plt.show()
    
with torch.no_grad():  
    radius = 1
    dist = 1
    idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
    idx_ch =  int(np.random.randint(0, 22, 1))
    
    x = tmp_dataset[idx_trial][0].unsqueeze(0)
    
    # Original amplitude
    output = model(x)
    x_r = output[0]
    distance, _ = fastdtw(x[0,0,idx_ch], x_r[0,0,idx_ch], radius = radius, dist = dist)
    print("Distance (original):\t\t", distance)
    plot_quick(x[0,0,idx_ch], x_r[0,0,idx_ch])
    
    # Standardization
    x = (x - x.mean())/x.std()
    output = model(x)
    x_r = output[0]
    x_r = (x_r - x_r.mean())/x_r.std()
    distance, _ = fastdtw(x[0,0,idx_ch], x_r[0,0,idx_ch], radius = radius, dist = dist)
    print("Distance (standardization):\t", distance)
    plot_quick(x[0,0,idx_ch], x_r[0,0,idx_ch])
    
    # Standardization
    x = (x - x.min())/(x.max() - x.min())
    output = model(x)
    x_r = output[0]
    x_r = (x_r - x_r.min())/(x_r.max() - x_r.min())
    distance, _ = fastdtw(x[0,0,idx_ch], x_r[0,0,idx_ch], radius = radius, dist = dist)
    print("Distance (minmax):\t\t\t", distance)
    plot_quick(x[0,0,idx_ch], x_r[0,0,idx_ch])
#%%
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtw_distance_matrix_train_2_train = model.dtw_comparison_2(train_dataset[:][0], device)
dtw_distance_matrix_train_2_test = model.dtw_comparison_2(test_dataset[:][0],  device)