import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal

import moabb.datasets as mb
import moabb.paradigms as mp
import config_dataset as cd
import config_plot as cp

"""
%load_ext autoreload
%autoreload 2
"""

def plot_random_trial_time_domain(trials_matrix, ch_list, config : dict):
    """
    Select n_trials randomly for a specific channel and plot them (in time domain)

    trials_matrix = numpy array with the EEG data. The shape must be N x C x T with N = number of trials, C = number of channels, T = time samples 
    ch_list = numpy array with the name of all the channels
    """
    
    # Get the index of the channel
    idx_ch_to_plot = ch_list == config['ch_to_plot']
    
    # Numpy array with the indices of all trials
    idx_all_trials = np.arange(trials_matrix.shape[0])

    # Sample random trials to plot
    idx_trial_to_plot = np.int32(np.random.choice(idx_all_trials, config['n_trials_to_plot'], replace = False))
    trials_to_plot = trials_matrix[idx_trial_to_plot, idx_ch_to_plot]

    # Create time vector
    t = np.linspace(config['t_start'], config['t_end'], trials_to_plot.shape[-1])
    
    # Create figure
    fig, ax = plt.subplots(1,1, figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})

    # Plot the signal
    for i in range(config['n_trials_to_plot']):
        ax.plot(t, trials_to_plot[i], label = "Trial n.{}".format(idx_trial_to_plot[i]))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EEG Signal")
    ax.set_xlim([t[0], t[-1]])
    
    # Remove empty space and add legend
    fig.legend()
    fig.tight_layout()
    
    if config['save_plot']: 
        fig.savefig("Plot/eeg_time_domain_subject_{}_{}.png".format(config['subject'], config['ch_to_plot']))
    
    if config['show_fig']: fig.show()

def visualize_single_subject_average_channel_ERS(stft_data, label_list, ch_list, config):
    """
    Visualize

    stft_data =  Numpy array with the ERS of EEG data. 
            The shape must be N x C x F x T with N = number of trials, C = number of channels, F = frequencies bins of stft, T = time samples of stft
    ch_list = list of length ch with the name of the channels
    label_list = numpy array of length N with the list of the label for each trial
    """
    
    # Compute the average data for the class-channels that I want to plot
    extracted_data = extract_data_to_plot(stft_data, ch_list, label_list, config)
    
    # Get the array for x-axis and y-axis
    t = config['t']
    f = config['f']
    
    # Create figure
    fig, ax = plt.subplots(len(config['label_to_plot']), len(config['ch_to_plot']), figsize = config['figsize'])
    plt.rcParams.update({'font.size': config['fontsize']})
    
    # Name of the class labels
    label_legend = {1 : 'Left', 2 : 'Right', 3 : 'Foot'}

    for i in range(len(config['label_to_plot'])):
        for j in range(len(config['ch_to_plot'])):
            # ax[i, j].plot(extracted_data[i][j])
            
            # Plot the stft
            im = ax[i, j].pcolormesh(t, f, extracted_data[i][j], shading='gouraud')
            im = ax[i, j].pcolormesh(t, f, extracted_data[i][j], shading='gouraud', vmin = np.min(extracted_data[i][j]), vmax = np.max(extracted_data[i][j]))

            # Label, title, limit  for each subplot
            ax[i, j].set_ylabel('Frequency [Hz]')
            ax[i, j].set_xlabel('Time [sec]')
            ax[i, j].title.set_text("{} - {}".format(label_legend[config['label_to_plot'][i]], config['ch_to_plot'][j]))
            if 'y_limit' in config: ax[i, j].set_ylim(config['y_limit'])
    
            fig.colorbar(im)

    # Title for the figure and remove empty space
    fig.suptitle('Subject {}'.format(config['subject']))
    fig.tight_layout()

    if config['save_plot']: 
        fig.savefig("Plot/ERS_Subject_{}.png".format(config['subject']))

    # Visualize figure
    if config['show_fig']: fig.show()

def extract_data_to_plot(data, ch_list, label_list, config):
    """
    Function that for each class and each channel compute the average trial (e.g. the average trial of class 'foot' for channel C3)
    """
    # List of eeg data with specific channels and average along the class
    extracted_data = []

    for label in config['label_to_plot']:
        # Get the indices of all the trial for a specific label
        idx_label = label_list == label

        # Get all the trial of the specific label and do the mean across the trial
        # The results is a matrix of shape C x T that contain the average accross the trial for a specific class
        average_per_label = data[idx_label].mean(0)
        
        # Get the channels that I want to plot
        tmp_list = []
        for ch in config['ch_to_plot']:
            idx_ch = ch_list == ch
            tmp_list.append(average_per_label[idx_ch].squeeze())
        
        # Save the list of data for the channels that I want to plot for this specific label
        extracted_data.append(tmp_list)

    return extracted_data 

def plot_average_band_stft(stft_data, ch_list, freq_array, config):
    # Get the index of the channel
    idx_ch_to_plot = ch_list == config['ch_to_plot']
    
    # Sample random trials to plot
    idx_all_trials = np.arange(stft_data.shape[0])
    idx_trial_to_plot = np.int32(np.random.choice(idx_all_trials, config['n_trials_to_average'], replace = False))
    sampled_stft = stft_data[idx_trial_to_plot, idx_ch_to_plot]
    
    # Compute the average for the specified band
    average_band_stft = compute_average_band_stft(sampled_stft, freq_array, config)

    # Create time vector
    t = np.linspace(config['t_start'], config['t_end'], len(average_band_stft))

    fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
    
    ax.plot(t, average_band_stft)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("STFT Average {}-{} Hz".format(config['band_start'], config['band_end']))
    ax.set_xlim([t[0], t[-1]])

    ax.set_title("Subject {} - Channel {} - Average {} trials".format(config['subject'], config['ch_to_plot'], config['n_trials_to_average']))
    
    fig.tight_layout()
    if config['save_plot']: 
        fig.savefig("Plot/Average_stft_Subject_{}_{}.png".format(config['subject'], config['ch_to_plot']))
    
    # Show figure
    if config['show_fig']: fig.show()

def compute_average_band_stft(stft_data, freq_array, config):
    # Average accross the trials
    average_stft_data = stft_data.mean(0)
    # print(average_stft_data.shape)

    # Select the indices of the frequency band I'm interested
    idx_band = np.logical_and(freq_array >= config['band_start'], freq_array <= config['band_end'])
    average_band_stft = average_stft_data[idx_band, :].mean(0)
    
    return average_band_stft 

def show_filter_effect_on_trial(trials_matrix, ch_list, filter_config, plot_config):
    """
    Function that show the effect of filtering on a random trial
    """
    
    # Get the eeg trial
    if 'n_trial_to_plot' not in plot_config: idx_trial = np.random.randint(0, trials_matrix.shape[0])
    else: idx_trial = plot_config['n_trial_to_plot']
    eeg_trial = trials_matrix[idx_trial].copy()
   
    # Additional info to convert the trial in RawArray of mne library
    info = mne.create_info(
        ch_names = list(ch_list), ch_types=["eeg"] * len(ch_list), sfreq = filter_config['sampling_freq']
    )

    # Create RawArray
    eeg_trial_RawArray = mne.io.RawArray(eeg_trial.copy(), info)

    # Filter the data and get back the numpy array
    eeg_trial_RawArray = filter_RawArray(eeg_trial_RawArray, filter_config)
    eeg_trial_filtered = eeg_trial_RawArray.get_data()
    
    # Get the data of the channel I want to plot
    idx_ch = ch_list == plot_config['ch_to_plot']
    eeg_signal = eeg_trial[idx_ch].squeeze()
    eeg_signal_filtered = eeg_trial_filtered[idx_ch].squeeze()

    # Create time vector
    t = np.linspace(plot_config['t_start'], plot_config['t_end'], len(eeg_signal))

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax.plot(t, eeg_signal, label = "Original signal")
    ax.plot(t, eeg_signal_filtered, label = "Filtered signal")

    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("EEG signal")
    
    filter_string = construct_filter_string(filter_config)
    ax.set_title("Subject {} - Channel {} - {}".format(plot_config['subject'], plot_config['ch_to_plot'], filter_string))

    fig.legend()
    fig.tight_layout()

    if plot_config['save_plot']: 
        fig.savefig("Plot/Filter_Effect_Subject_{}_{}_{}.png".format(plot_config['subject'], plot_config['ch_to_plot'], idx_trial))

    if plot_config['show_fig']: fig.show()


def construct_filter_string(filter_config):
    filter_string = ""
    
    filter_string += filter_config['filter_method'].upper()

    if filter_config['filter_type'] == 0: # Bandpass
        filter_string += " Bandpass" 
        filter_string += " {}Hz-{}Hz".format(filter_config['fmin'], filter_config['fmax'])
    if filter_config['filter_type'] == 1: # Lowpass
        filter_string += " Lowpass" 
        filter_string += " {}Hz".format(filter_config['fmax'])
    if filter_config['filter_type'] == 2: # Highpass 
        filter_string += " Highpass" 
        filter_string += " {}Hz".format(filter_config['fmin'])

    if filter_config['filter_method'] == 'iir':
        filter_string += " Order {}".format(filter_config['iir_params']['order'])

    return filter_string

def show_after_before_ERS_on_trial(trials_matrix_before_ERS, trials_matrix_after_ERS, ch_list, f, t, config):
    # Indices of trial and channel to plot
    if 'n_trial_to_plot' not in config: idx_trial = np.random.randint(0, trials_matrix_before_ERS.shape[0])
    else: idx_trial = config['n_trial_to_plot']
    idx_ch_to_plot = ch_list == config['ch_to_plot']

    trial_before = trials_matrix_before_ERS[idx_trial, idx_ch_to_plot].squeeze()
    trial_after = trials_matrix_after_ERS[idx_trial, idx_ch_to_plot].squeeze()

    fig, ax = plt.subplots(1, 2, figsize = config['figsize'])

    im = ax[0].pcolormesh(t, f, trial_before, shading = 'gouraud', cmap = config['cmap'])
    fig.colorbar(im)
    im = ax[1].pcolormesh(t, f,trial_after, shading = 'gouraud', cmap = config['cmap'])
    fig.colorbar(im)

    fig.tight_layout()
    fig.show()
