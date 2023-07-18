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
import scipy.signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
import library.train_generic as train_generic

#%%

dataset_config = cd.get_moabb_dataset_config([3])
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

epoch = 'BEST'
path_weight = 'TMP_Folder/model_{}.pth'.format(epoch)


model.load_state_dict(torch.load(path_weight))


with torch.no_grad():
    for i in range(1):
        idx_trial = int(np.random.randint(0, len(train_dataset), 1))
        idx_ch =  int(np.random.randint(0, 22, 1))
        idx_trial = 251
        idx_ch = 9
        x = train_dataset[idx_trial][0]
        x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x.unsqueeze(0))
        
        # f, x_r_psd = signal.welch(tmp_x_r_ch, fs = fs, nperseg = nperseg, noverlap = noverlap)
        
        t = np.linspace(2, 7, x.shape[-1])
        idx_t = np.logical_and(t > 2.2, t < 6.8)
        idx_t = np.ones(len(t)) == 1
        
        x_plot = x.squeeze()[idx_ch]
        x_r_plot = x_r.squeeze()[idx_ch, idx_t]
        
        x_plot = (x_plot - x_plot.mean())/x_plot.std()
        x_r_plot = (x_r_plot - x_r_plot.mean())/x_r_plot.std()
        
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t, x_plot)
        plt.plot(t[idx_t], x_r_plot)
        plt.xlabel("Time [s]")
        plt.title("Ampiezza originale")
        plt.show()