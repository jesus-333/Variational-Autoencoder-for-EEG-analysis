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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def visualize_set_of_trials(data, config : dict, x_r = None):
    """
    Concatenate a series of trial from the datset and plot them
    """
    n_trial = config['idx_end'] - config['idx_start']
    x_plot = extract_and_flat_trial(data, config)
    t_plot = np.linspace(0, n_trial * config['trial_length'], len(x_plot))

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})
    
    # Plot signals
    ax.plot(t_plot, x_plot, label = 'Original signal')
    if x_r is not None: # If you want to plot also the reconstructed signal
        x_r_plot = extract_and_flat_trial(x_r, config)
        ax.plot(t_plot, x_r_plot, label = 'Reconstructed signal')
        ax.legend()
    
    # "Beautify" plot
    if config['add_trial_line']:
        for i in range(n_trial): ax.axvline((i + 1) * config['trial_length'], color = 'red')
        ax.grid(axis = 'y')
    else:
        ax.grid(True)
        
    ax.set_xlim([t_plot[0], t_plot[-1]])
    ax.set_xlabel("N. trials")
    ax.set_xticks(ticks = (np.arange(n_trial) * 4) + 2, labels = np.arange(n_trial) + config['idx_start'])
    
    ax.set_ylabel("Amplitde [microV]")
    
    # Show plot
    fig.tight_layout()
    fig.show()
    

def extract_and_flat_trial(data, config):
    x = data[config['idx_start']:config['idx_end']]
    x_ch = x[:, 0, config['idx_ch'], :] # The dimension with 0 is the depth
    x_plot = x_ch.flatten()
    
    return x_plot


def plot_latent_space_embedding(z, config : dict, color = None):
    colormap = config['colormap'] if 'colormap' in config else 'viridis'
    markersize = config['markersize'] if 'markersize' in config else 1

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})
    
    # Plot the embedding
    im = ax.scatter(x = z[:, 0], y = z[:, 1], s = markersize,
               c = color, cmap = colormap
                )
    
    # Extra stuff
    fig.colorbar(im, ax=ax)
    ax.grid(True)
    fig.tight_layout()
    fig.show()

# subj = [2]
#
# dataset_config = cd.get_moabb_dataset_config(subj)
# device = 'cpu'
#
# C = 22
# if dataset_config['resample_data']: sf = dataset_config['resample_freq']
# else: sf = 250
# T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
# type_decoder = 0
# parameters_map_type = 0
# model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)
#
# train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)    
#
# # Create model
# model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
# model = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)
# model.to(device)
#
# #%%
#
# tmp_dataset = test_dataset
#
# epoch = 'BEST'
# path_weight = 'TMP_Folder/model_{}.pth'.format(epoch)
#
# # path_weight = 'TMP_Folder/model_BEST_2.pth'
# # path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_3_not_normalized/model_BEST.pth'
# path_weight = 'TMP_Folder/Perfect_recon_dwt_subj_5_not_normalized/model_BEST.pth'
#
# model.load_state_dict(torch.load(path_weight))
#
# # subj 3 idx --> label  
# # 12 --> 0 (left), 33 --> 1 (right), 9 --> 2 (foot) 254 --> 3 (tongue)
#
# label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'togue' }
# label_to_ch = {'left' : 11, 'right' : 7, 'foot' : 9, 'togue' : -1 }
#
# with torch.no_grad():
#     for i in range(33):
#         # idx_trial = 33
#         # idx_ch = 7
#         # idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
#         # idx_ch =  int(np.random.randint(0, 22, 1))
#         # x = tmp_dataset[idx_trial][0]
#         
#         idx_trial = int(np.random.randint(0, len(tmp_dataset), 1))
#         x = tmp_dataset[idx_trial][0]
#         label = label_dict[int(tmp_dataset[idx_trial][1])]
#         if label == 'tongue': pass
#         idx_ch = label_to_ch[label]
#         
#         output = model(x.unsqueeze(0))
#         x_r = output[0]
#                 
#         
#         t = np.linspace(2, 7, x.shape[-1])
#         idx_t = np.logical_and(t > 2, t < 6)
#         # idx_t = np.ones(len(t)) == 1
#         
#         x_plot = x.squeeze()[idx_ch, idx_t]
#         x_r_plot = x_r.squeeze()[idx_ch, idx_t]
#         
#         # x_plot = (x_plot - x_plot.mean())/x_plot.std()
#         # x_r_plot = (x_r_plot - x_r_plot.mean())/x_r_plot.std()
#         
#         plt.rcParams.update({'font.size': 20})
#         plt.figure(figsize = (15, 10))
#         plt.plot(t[idx_t], x_plot, label = 'Original Signal')
#         plt.plot(t[idx_t], x_r_plot, label = 'Reconstructed Signal')
#         plt.xlabel("Time [s]")
#         plt.title("Subj {} - Trial {} - Label: {} - Ch: {}".format(subj[0], idx_trial, label, tmp_dataset.ch_list[idx_ch]))
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#         
#         # f, x_psd = signal.welch(x_plot, fs = 250,)
#         # f, x_r_psd = signal.welch(x_r_plot * 20, fs = 250,)
#         # plt.rcParams.update({'font.size': 20})
#         # plt.figure(figsize = (15, 10))
#         # plt.plot(f, x_psd)
#         # plt.plot(f, x_r_psd)
#         # plt.xlabel("Frequency [Hz]")
#         # plt.title("Ampiezza originale")
#         # plt.show()
