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

#%%

subj = [3]

dataset_config = cd.get_moabb_dataset_config(subj)
device = 'cpu'

C = 22
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
type_decoder = 0
parameters_map_type = 0
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)

train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)    

# Create model
model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
model = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
model.to(device)

#%%

tmp_dataset = test_dataset

epoch = 'BEST'
path_weight = 'TMP_Folder/model_{}.pth'.format(epoch)
path_weight = 'Saved model/hvae_dwt/subj_3_100_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_1_30_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_2_20_{}.pth'.format(epoch)

# path_weight = 'TMP_Folder/model_BEST_2.pth'
# path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_3_not_normalized/model_BEST.pth'
# path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_5_not_normalized/model_BEST.pth'

model.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))

# subj 3 idx --> label  
# 12 --> 0 (left), 33 --> 1 (right), 9 --> 2 (foot) 254 --> 3 (tongue)

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
label_to_ch = {'left' : 11, 'right' : 7, 'foot' : 9, 'tongue' : -1 }

with torch.no_grad():
    for i in range(20):
        # idx_trial = 50
        # idx_ch = 11
        # idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        # idx_ch =  int(np.random.randint(0, 22, 1))
        # x = tmp_dataset[idx_trial][0]
        
        idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        x = tmp_dataset[idx_trial][0]
        label = label_dict[int(tmp_dataset[idx_trial][1])]
        if label == 'tongue': pass
        idx_ch = label_to_ch[label]
        
        output = model(x.unsqueeze(0))
        x_r = output[0]
                  
        t = np.linspace(2, 7, x.shape[-1])
        idx_t = np.logical_and(t > 4, t < 6)
        # idx_t = np.ones(len(t)) == 1
        
        x_plot = x.squeeze()[idx_ch, idx_t]
        x_r_plot = x_r.squeeze()[idx_ch, idx_t]
        
        # x_plot = (x_plot - x_plot.mean())/x_plot.std()
        # x_r_plot = (x_r_plot - x_r_plot.mean())/x_r_plot.std()
        
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t[idx_t], x_plot, label = 'Original Signal')
        plt.plot(t[idx_t], x_r_plot, label = 'Reconstructed Signal')
        plt.xlabel("Time [s]")
        plt.title("Subj {} - Trial {} - Label: {} - Ch: {}".format(subj[0], idx_trial, label, tmp_dataset.ch_list[idx_ch]))
        plt.legend()
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
        
#%% Test generation different subject

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