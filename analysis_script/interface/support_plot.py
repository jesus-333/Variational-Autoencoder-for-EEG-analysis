"""
Contains functions used to load data

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import support_data
import support_variables

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_both_error_map_and_eeg_signal_plots(streamlit_container_for_the_plot_error_map, streamlit_container_for_the_plot_eeg_signal) :
    create_recon_error_map(streamlit_container_for_the_plot_error_map)
    create_eeg_singal_plot(streamlit_container_for_the_plot_eeg_signal)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_recon_error_map(streamlit_container_for_the_plot) :
    
    subj = st.session_state.subj_selection.split(' ')[1]
    session = st.session_state.session_selection.lower()
    epoch = st.session_state.epoch_selection

    path_recon_error_file = f"./data/recon_error_d2a/S{subj}/{session}/recon_error_{epoch}_average.npy"

    data_train = support_data.load_recon_error_file(path_recon_error_file)

    width = st.session_state.figsize_width_error_map if 'figsize_width_error_map' in st.session_state else 15
    height = st.session_state.figsize_height_error_map if 'figsize_height_error_map' in st.session_state else 15
    
    plot_config = dict(
        figsize = (width, height),
        min_value = st.session_state.min_value if 'min_value' in st.session_state else 0,
        max_value = st.session_state.max_value if 'max_value' in st.session_state else 55,
        fontsize = st.session_state.fontsize_error_map if 'fontsize_error_map' in st.session_state else 15,
        colormap = st.session_state.colormap if 'colormap' in st.session_state else 'Reds',
        add_runs_grid = st.session_state.add_runs_grid if 'add_runs_grid' in st.session_state else True,
        add_trials_grid = st.session_state.add_trials_grid if 'add_trials_grid' in st.session_state else False,
        add_colorbar = st.session_state.add_colorbar if 'add_colorbar' in st.session_state else False,
        transpose_error_map = st.session_state.transpose_error_map if 'transpose_error_map' in st.session_state else False,
        visualize_trial_error_map = st.session_state.visualize_trial_error_map if 'visualize_trial_error_map' in st.session_state else False,
        visualize_trial_error_map_mode = st.session_state.visualize_trial_error_map_mode if 'visualize_trial_error_map_mode' in st.session_state else 'Circle',
    )

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    plot_recon_error_map(data_train, fig, ax, plot_config)

    with streamlit_container_for_the_plot :
        st.pyplot(fig, width = 'content')


def plot_recon_error_map(data, fig, ax, plot_config : dict) :

    data_to_plot = data.T if plot_config['transpose_error_map'] else data

    # Plot image
    img = ax.imshow(data_to_plot, cmap = plot_config['colormap'], aspect = 'auto',
                    vmin = plot_config['min_value'], vmax = plot_config['max_value']
                    )

    # Add labels to axis
    if plot_config['transpose_error_map']:
        ax.set_xlabel('Repetitions', fontsize = plot_config['fontsize'])
        ax.set_ylabel('Channels', fontsize = plot_config['fontsize'])
        
        # Set ticks for the run/trials
        xticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48
        ax.set_xticks(xticks - 0.5, labels = xticks, fontsize = plot_config['fontsize'])
        if plot_config['add_trials_grid'] : ax.set_xticks(np.arange(288) - 0.5, minor = True, labels = [])

        # Set ticks for the channels
        yticks = np.arange(22)
        ax.set_yticks(yticks - 0.5, labels = [])
        ax.set_yticks(yticks, labels = support_variables.CHANNELS_D2A, minor = True, fontsize = plot_config['fontsize'])
        ax.tick_params(axis = "y", bottom = True, top = True, labelbottom = True, labeltop = True)
        ax.tick_params(axis = "y", which = 'minor', left = True, right = True, labelleft = True, labelright = True)
    else:
        ax.set_ylabel('Repetitions', fontsize = plot_config['fontsize'])
        ax.set_xlabel('Channels', fontsize = plot_config['fontsize'])

        # Set ticks for the run/trials
        yticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48
        ax.set_yticks(yticks - 0.5, labels = yticks, fontsize = plot_config['fontsize'])
        if plot_config['add_trials_grid'] : ax.set_yticks(np.arange(288) - 0.5, minor = True, labels = [])

        # Set ticks for the channels
        xticks = np.arange(22)
        ax.set_xticks(xticks - 0.5, labels = [])
        ax.set_xticks(xticks, labels = support_variables.CHANNELS_D2A, minor = True, fontsize = plot_config['fontsize'])
        ax.tick_params(axis = "x", which = 'minor', bottom = True, top = True, labelbottom = True, labeltop = True)

    # if plot_config['add_title'] : ax.set_title('Subject {}'.format(subj), fontsize = plot_config['fontsize'] + 2)
    # if plot_config['add_title'] : ax.set_title('Session {}'.format(i + 1), fontsize = plot_config['fontsize'] + 2)
    
    if plot_config['visualize_trial_error_map'] :
        if plot_config['visualize_trial_error_map_mode'] == 'Circle' :
            highlisht_selected_trial_circle(plot_config, ax)
        elif plot_config['visualize_trial_error_map_mode'] == 'Cross' :
            highlisht_selected_trial_cross(plot_config, ax)
        elif plot_config['visualize_trial_error_map_mode'] == 'Both' :
            highlisht_selected_trial_circle(plot_config, ax)
            highlisht_selected_trial_cross(plot_config, ax)
    
    # Add grid
    if plot_config['add_runs_grid'] and not plot_config['add_trials_grid'] :
        ax.grid(True, color = 'black', which = 'major')
    elif not plot_config['add_runs_grid'] and plot_config['add_trials_grid'] :
        ax.grid(True, color = 'grey', which = 'minor', axis = 'x' if plot_config['transpose_error_map'] else 'y')
    elif plot_config['add_runs_grid'] and plot_config['add_trials_grid'] :
        ax.grid(True, color = 'black', which = 'major', axis = 'both')
        ax.grid(True, color = 'grey', which = 'minor', axis = 'x' if plot_config['transpose_error_map'] else 'y')

    if plot_config['add_colorbar']:
        fig.colorbar(img)

def highlisht_selected_trial_circle(plot_config : dict, ax : plt.Axes) :
    """
    Highlight the selected trial in the error map drawing a circle around the corresponding point.
    """

    x_coord, y_coord = get_coord_selected_trial(plot_config)

    circle = plt.Circle((x_coord, y_coord), 0.6, color = 'blue', fill = False, linewidth = 2)
    ax.add_patch(circle)

def highlisht_selected_trial_cross(plot_config : dict, ax : plt.Axes) :
    """
    Highlight the selected trial in the error map drawing a line in the corresponding row and column.
    """

    x_coord, y_coord = get_coord_selected_trial(plot_config)

    ax.axhline(y_coord, color = 'blue')
    ax.axvline(x_coord, color = 'blue')

def get_coord_selected_trial(plot_config : dict) :
    """
    Get the coordinate inside the error map figure of the selected trial
    """

    channel     = st.session_state.channel
    channel_idx = np.where(support_variables.CHANNELS_D2A == channel)[0][0]
    trial_idx   = st.session_state.trials - 1

    if plot_config['transpose_error_map'] :
        x_coord = trial_idx
        y_coord = channel_idx
    else :
        x_coord = channel_idx
        y_coord = trial_idx

    return x_coord, y_coord

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def create_eeg_singal_plot(streamlit_container_for_the_plot) :
    subj = st.session_state.subj_selection.split(' ')[1]
    session = st.session_state.session_selection.lower()
    epoch = st.session_state.epoch_selection
    trial = st.session_state.trials - 1

    path_recon_error_file = f"./data/d2a_dataset/S{subj}/{session}/trial_{trial}.npy"
    data = support_data.load_eeg_data_file(path_recon_error_file)
    
    # Get domain
    domain = st.session_state.domain_eeg_signal if 'domain_eeg_signal' in st.session_state else 'time'
    
    # Get width and height of the figure
    width = st.session_state.figsize_width_eeg_signal if 'figsize_width_eeg_signal' in st.session_state else 15
    height = st.session_state.figsize_height_eeg_signal if 'figsize_height_eeg_signal' in st.session_state else 7

    # When domain is 'Both', the figure will contain two subplots side by side
    # Widht is doubled to keep the aspect ratio
    if domain == 'Both' : width = width * 2

    plot_config = dict(
        figsize = (width, height),
        fontsize = st.session_state.fontsize_eeg_plot if 'fontsize_eeg_plot' in st.session_state else 17,
        domain_eeg_signal = domain,
        color_original = st.session_state.color_original_eeg_plot if 'color_original_eeg_plot' in st.session_state else 'black',
        linewidth_original = st.session_state.linewidth_original_eeg_plot if 'linewidth_original_eeg_plot' in st.session_state else 2,
        color_reconstructed = st.session_state.color_reconstructed_eeg_plot if 'color_reconstructed_eeg_plot' in st.session_state else 'red',
        linewidth_reconstructed = st.session_state.linewidth_reconstructed_eeg_plot if 'linewidth_reconstructed_eeg_plot' in st.session_state else 2,
        rescale_minmax = st.session_state.rescale_minmax_eeg_plot if 'rescale_minmax_eeg_plot' in st.session_state else False,
        add_grid = st.session_state.add_grid_eeg_plot if 'add_grid_eeg_plot' in st.session_state else True,
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Time domain specific parameters

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Frequency domain specific parameters
        nperseg = st.session_state.nperseg if 'nperseg' in st.session_state else 500,
    )
    
    # Get the data to plot
    channel_idx = np.where(support_variables.CHANNELS_D2A == st.session_state.channel)[0][0]
    x_eeg_time = data[channel_idx, :]

    if plot_config['domain_eeg_signal'] == 'Time' :
        # Time domain
        t = np.arange(x_eeg_time.shape[0]) / 250 + 2
        # The dataset d2a contains trials of 6s sampled at 250Hz. Since the first 2s are baseline, I shift the time axis by 2s.
        # Note that I previoulsy downloaded and stored them locally, removing the first 2s of baseline.

        fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
        plot_eeg_signal_time(t, x_eeg_time, fig, ax, plot_config)

    elif plot_config['domain_eeg_signal'] == 'Frequency' :
        # Frequency domain
        f, x_eeg_freq = support_data.compute_spectra(x_eeg_time, plot_config['nperseg'])

        fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

        plot_eeg_signal_frequency(f, x_eeg_freq, fig, ax, plot_config)

    elif plot_config['domain_eeg_signal'] == 'Both' :
        # Both time and frequency domain
        t = np.arange(x_eeg_time.shape[0]) / 250 + 2
        f, x_eeg_freq = support_data.compute_spectra(x_eeg_time, plot_config['nperseg'])

        fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize = plot_config['figsize'])

        plot_eeg_signal_time(t, x_eeg_time, fig, ax_time, plot_config)
        plot_eeg_signal_frequency(f, x_eeg_freq, fig, ax_freq, plot_config)
    
    # Display the figure in streamlit
    with streamlit_container_for_the_plot :
        st.pyplot(fig, width = 'content')


def plot_eeg_signal_time(t : np.ndarray, x_eeg_time : np.ndarray, fig : plt.Figure, ax : plt.Axes, plot_config : dict) :
    """

    """

    # Optional rescale
    if plot_config['rescale_minmax'] :
        x_eeg_time = (x_eeg_time - x_eeg_time.min()) / (x_eeg_time.max() - x_eeg_time.min())

    ax.plot(t, x_eeg_time, label = 'EEG signal (time)',
            color = plot_config['color_original'], linewidth = plot_config['linewidth_original']
            )
    
    ax.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
    if plot_config['rescale_minmax'] :
        ax.set_ylabel(r"Normalized amplitude", fontsize = plot_config['fontsize'])
    else :
        ax.set_ylabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])

    beautify_plot(fig, ax, plot_config, t)
    

def plot_eeg_signal_frequency(f : np.ndarray, x_eeg_time : np.ndarray, fig : plt.Figure, ax : plt.Axes, plot_config : dict) :
    """

    """

    # Optional rescale
    if plot_config['rescale_minmax'] :
        x_eeg_time = (x_eeg_time - x_eeg_time.min()) / (x_eeg_time.max() - x_eeg_time.min())

    ax.plot(f, x_eeg_time, label = 'EEG signal (frequency)',
            color = plot_config['color_original'], linewidth = plot_config['linewidth_original']
            )
    
    ax.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
    if plot_config['rescale_minmax'] :
        ax.set_ylabel(r"Normalized Power Spectral Density", fontsize = plot_config['fontsize'])
    else :
        ax.set_ylabel(r"Power Spectral Density [$\mu$V$^2$/Hz]", fontsize = plot_config['fontsize'])

    beautify_plot(fig, ax, plot_config, f)


def beautify_plot(fig : plt.Figure, ax : plt.Axes, plot_config : dict, x_axis_values : np.ndarray) :
    """
    Beautify the plot according to the given configuration.
    """

    if plot_config['add_grid'] :
        ax.grid(True)
    
    ax.tick_params(axis = 'both', labelsize = plot_config['fontsize'])
    ax.legend(fontsize = plot_config['fontsize'])

    ax.set_xlim([x_axis_values[0], x_axis_values[-1]])

    fig.tight_layout()

