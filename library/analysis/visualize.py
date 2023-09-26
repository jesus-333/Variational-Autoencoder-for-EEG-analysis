# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:06:56 2023

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

import numpy as np
import torch 
import matplotlib.pyplot as plt
import scipy.signal as signal

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def plot_single_eeg_channel(x_eeg, t, config):
    if 'plot_psd' in config: 
        if config['plot_psd']:
            if 'nperseg' not in config: config['nperseg'] = 256
            if 'fs' not in config: config['fs'] = 1
            t, x_eeg = signal.welch(x_eeg, fs = config['fs'], nperseg =  config['nperseg'])
    else:
        config['plot_psd'] = False
    
    if 'fontsize' in config: plt.rcParams.update({'font.size': config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    
    ax.plot(t, x_eeg)
    
    ax.set_xlim([t[0], t[-1]])
    if config['plot_psd']: 
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")
    else:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("EEG Amplitude")
    if 'title' in config: ax.set_title(config['title'])
    ax.grid(True)
    
    fig.tight_layout()
    fig.show()
    
    return fig, ax


def plot_single_eeg_channel_both_time_and_frequency(x_eeg, t, config):

    if 'nperseg' not in config: config['nperseg'] = 256
    if 'fs' not in config: config['fs'] = 1
    f, x_eeg_psd = signal.welch(x_eeg, fs = config['fs'], nperseg =  config['nperseg'])
    
    if 'fontsize' in config: plt.rcParams.update({'font.size': config['fontsize']})
    fig, ax = plt.subplots(1, 2, figsize = config['figsize'])
    
    ax[0].plot(t, x_eeg)
    ax[1].plot(f, x_eeg_psd)
    
    ax[0].set_xlim([t[0], t[-1]])
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("EEG Amplitude")
    
    ax[1].set_xlim([f[0], f[-1]])
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("PSD")

    if 'title' in config: fig.suptitle(config['title'])
    ax[0].grid(True)
    ax[1].grid(True)
    
    fig.tight_layout()
    fig.show()
    
    return fig, ax

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
