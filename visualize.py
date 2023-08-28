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
import os

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.config import config_plot as cp
from library.dataset import preprocess as pp
from library.analysis import latent_space
import library.training.train_generic as train_generic
import library.analysis.visualize as visualize


#%% Load data

subj = [3]
subj = [1,2,3,4,5,6,7,8,9]

dataset_config = cd.get_moabb_dataset_config(subj)
device = 'cpu'

C = 22
if dataset_config['resample_data']: sf = dataset_config['resample_freq']
else: sf = 250
T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
type_decoder = 0
parameters_map_type = 0

train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Create model (hvEEGNet)
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
model_config['use_classifier'] = False
model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
model_hv.to(device)

# Create model (vEEGNet)
model_config = cm.get_config_vEEGNet(C, T, 64, type_decoder = 0, type_encoder = 0)
model_config['use_classifier'] = False
model_config['parameters_map_type'] = 0
model_config['encoder_config']['p_kernel_1'] = None
model_config['encoder_config']['p_kernel_2'] = (1, 10)
model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
model_v = train_generic.get_untrained_model('vEEGNet', model_config)
model_v.to(device)

#%% Test RECONSTRUCTION

tmp_dataset = test_dataset

epoch = 'BEST'
# path_weight = 'TMP_Folder/model_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_3_100_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_1_30_{}.pth'.format(epoch)
# path_weight = 'Saved model/hvae_dwt/subj_2_20_{}.pth'.format(epoch)

# path_weight = 'TMP_Folder/model_BEST_2.pth'
path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_3_not_normalized/model_BEST.pth'
path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_5_not_normalized/model_BEST.pth'


# model_hv.load_state_dict(torch.load('Saved Model/full_eeg/{}/model_BEST.pth'.format(subj[0]), map_location=torch.device('cpu')))
# model_v.load_state_dict(torch.load('Saved Model/vEEGNet_dtw/{}/model_BEST.pth'.format(subj[0]), map_location=torch.device('cpu')))

model_hv.load_state_dict(torch.load(path_hv_weight, map_location=torch.device('cpu')))

# subj 3 idx --> label
# 12 --> 0 (left), 33 --> 1 (right), 9 --> 2 (foot) 254 --> 3 (tongue)

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
label_to_ch = {'left' : 7, 'right' : 11, 'foot' : 9, 'tongue' : -1 }

# trial 47 subj 3 ch C3

with torch.no_grad():
    for i in range(1):
        idx_trial = 47
        idx_ch = 7
        # idx_trial = 214
        # idx_ch = 0
        # idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        # idx_ch =  int(np.random.randint(0, 22, 1))
        # x = tmp_dataset[idx_trial][0]

        # idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
        x = tmp_dataset[idx_trial][0]
        label = label_dict[int(tmp_dataset[idx_trial][1])]
        if label == 'tongue': pass
        # idx_ch = label_to_ch[label]
        
        output = model_v(x.unsqueeze(0).to(device))
        x_r = output[0].to('cpu')
        
        t = np.linspace(2, 6, x.shape[-1])
        idx_t = np.logical_and(t >= 2, t <= 4)
        x_plot = x.squeeze()[idx_ch, idx_t]
        
        t_r = np.linspace(2, 6, x_r.shape[-1])
        idx_t_r = np.logical_and(t_r >= 2, t_r <= 4)
        x_r_plot = x_r.squeeze()[idx_ch, idx_t_r]

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t[idx_t], x_plot, label = 'Original Signal')
        plt.plot(t_r[idx_t_r], x_r_plot, label = 'Reconstructed Signal')
        plt.xlabel("Time [s]")
        plt.title("Subj {} - Trial {} - Label: {} - Ch: {}".format(subj[0], idx_trial, label, tmp_dataset.ch_list[idx_ch]))
        plt.legend()
        plt.ylim([-40, 40])
        plt.tight_layout()
        plt.grid(True)
         
        output = model_hv(x.unsqueeze(0).to(device))
        x_r = output[0].to('cpu')
        
        t = np.linspace(2, 6, x.shape[-1])
        idx_t = np.logical_and(t >= 2, t <= 4)
        x_plot = x.squeeze()[idx_ch, idx_t]
        
        t_r = np.linspace(2, 6, x_r.shape[-1])
        idx_t_r = np.logical_and(t_r >= 2, t_r <= 4)
        x_r_plot = x_r.squeeze()[idx_ch, idx_t_r]

        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize = (15, 10))
        plt.plot(t[idx_t], x_plot, label = 'Original Signal')
        plt.plot(t_r[idx_t_r], x_r_plot, label = 'Reconstructed Signal')
        plt.xlabel("Time [s]")
        plt.title("hvEEGNet_woclf - Subj {} - Trial {} - Label: {} - Ch: {}".format(subj[0], idx_trial, label, tmp_dataset.ch_list[idx_ch]))
        plt.legend()
        plt.ylim([-40, 40])
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # nperseg = 500
        # f, x_psd = signal.welch(x_plot, fs = 250, nperseg = nperseg)
        # f, x_r_psd = signal.welch(x_r_plot * 20, fs = 250, nperseg = nperseg)
        # plt.rcParams.update({'font.size': 20})
        # plt.figure(figsize = (15, 10))
        # plt.plot(f, x_psd, label = 'Original')
        # # plt.plot(f, x_r_psd, label = 'Recon')
        # plt.legend()
        # plt.xlabel("Frequency [Hz]")
        # plt.title("Ampiezza originale")
        # plt.show()
        
#%%

subj_list = [3, 5, 8]
loss_list = ['mse','dtw']
model_epoch_list = [20, 40, 'BEST']
plot_psd = True

idx_trial = 47
idx_ch = 7

t_min = 2
t_max = 4


def plot_figure(model, model_name, path_weight, dataset, subj, loss, model_epoch, plot_psd):
    label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
    label_to_ch = {'left' : 7, 'right' : 11, 'foot' : 9, 'tongue' : -1 }
    
    model.load_state_dict(torch.load(path_weight, map_location=torch.device('cpu')))
    
    x = dataset[idx_trial][0]
    label = label_dict[int(dataset[idx_trial][1])]
    
    output = model(x.unsqueeze(0).to(device))
    x_r = output[0].to('cpu')
    
    t = np.linspace(2, 6, x.shape[-1])
    idx_t = np.logical_and(t >= t_min, t <= t_max)
    x_plot = x.squeeze()[idx_ch, idx_t]

    t_r = np.linspace(2, 6, x_r.shape[-1])
    idx_t_r = np.logical_and(t_r >= t_min, t_r <= t_max)
    x_r_plot = x_r.squeeze()[idx_ch, idx_t_r]
    
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize = (15, 10))

    if plot_psd:
        nperseg = 500
        f, x_psd = signal.welch(x_plot, fs = 250, nperseg = nperseg)
        f_r, x_r_psd = signal.welch(x_r_plot, fs = 250, nperseg = nperseg)
        
        plt.plot(f, x_psd, label = 'Original Signal')
        plt.plot(f_r, x_r_psd, label = 'Reconstructed Signal')
        plt.xlabel("Frequency [Hz]")
    else:    
        plt.plot(t[idx_t], x_plot, label = 'Original Signal')
        plt.plot(t_r[idx_t_r], x_r_plot, label = 'Reconstructed Signal')
        plt.xlabel("Time [s]")
        plt.ylim([-40, 40])
        
    plt.title("{} - Subj {} - Trial {} - Label: {} - Ch: {} - Loss : {}".format(model_name, subj, idx_trial, label, dataset.ch_list[idx_ch], loss))
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
                

with torch.no_grad():
    for subj in subj_list:
        for loss in loss_list:
            for model_epoch in model_epoch_list:
    
                dataset_config = cd.get_moabb_dataset_config([subj])
                device = 'cpu'
                
                C = 22
                if dataset_config['resample_data']: sf = dataset_config['resample_freq']
                else: sf = 250
                T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
                type_decoder = 0
                parameters_map_type = 0
                
                train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
                                
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                #  vEEGNet
                
                model_config = cm.get_config_vEEGNet(C, T, 64, type_decoder = 0, type_encoder = 0)
                model_config['use_classifier'] = False
                model_config['parameters_map_type'] = 0
                model_config['encoder_config']['p_kernel_1'] = None
                model_config['encoder_config']['p_kernel_2'] = (1, 10)
                model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
                model_v = train_generic.get_untrained_model('vEEGNet', model_config)
                model_v.to(device)
                
                path_v_weight = 'Saved Model/vEEGNet_{}/{}/model_{}.pth'.format(loss, subj, model_epoch)
                model_name = 'vEEGNet_woclf'
                
                plot_figure(model_v, model_name, path_v_weight, test_dataset, subj, loss, model_epoch, plot_psd)
                
                path_save = 'Plot/vEEGNet_{}/{}/vEEGNet_woclf_eeg_reconstruction_trial_{}_ch_{}_epoch_{}.png'.format(loss, subj, idx_trial, idx_ch, model_epoch)
                if not os.path.isdir('Plot/vEEGNet_{}/{}'.format(loss, subj)): os.makedirs('Plot/vEEGNet_{}/{}'.format(loss, subj))
                plt.savefig(path_save)
                plt.show()
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # hvEEGNet
                
                model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
                model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
                model_config['use_classifier'] = False
                model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
                model_hv.to(device)
                
                path_hv_weight = 'Saved Model/hvEEGNet_shallow_{}/{}/model_{}.pth'.format(loss, subj, model_epoch)
                model_name = 'hvEEGNet_woclf'
                
                plot_figure(model_hv, model_name, path_hv_weight, test_dataset, subj, loss, model_epoch, plot_psd)
                
                path_save = 'Plot/hvEEGNet_shallow_{}/{}/hvEEGNet_woclf_eeg_reconstruction_trial_{}_ch_{}_epoch_{}.png'.format(loss, subj, idx_trial, idx_ch, model_epoch)
                if not os.path.isdir('Plot/hvEEGNet_shallow_{}/{}'.format(loss, subj)): os.makedirs('Plot/hvEEGNet_shallow_{}/{}'.format(loss, subj))
                plt.savefig(path_save)
                plt.show()
