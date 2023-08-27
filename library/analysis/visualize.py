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
