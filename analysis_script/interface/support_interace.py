"""
Contains functions used to build the interface

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import streamlit as st
import matplotlib

import support_plot
import support_variables

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_subject_selection(streamlit_container_error_map_plot, streamlit_container_eeg_signal) :
    epoch_array = (np.arange(16) + 1) * 5

    col_1, col_2, col_3 = st.columns([0.3, 0.3, 0.3], border = False)

    with col_1 :
        st.selectbox(
            label = 'Select the subject',
            options = support_variables.SUBJ_ARRAY, index = 1,
            key = 'subj_selection',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

    with col_2 :
        st.selectbox(
            label = 'Select the session',
            options = ['Train', 'Test'], index = 0,
            key = 'session_selection',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

    with col_3 :
        st.selectbox(
            label = 'Select the epoch',
            options = epoch_array, index = len(epoch_array) - 1,
            key = 'epoch_selection',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

def build_recon_error_options(streamlit_container_error_map_plot, streamlit_container_eeg_signal) :

    st.write("---")
    st.write("Options for the error map plot")

    option_col_1, option_col_2 = st.columns([0.5, 0.5], border = False, gap = "Large", vertical_alignment = "center")

    with option_col_1 :
        # Slider min_value recon error
        st.slider(
            label = "Value Min",
            help= "All value below this will have the same color",
            min_value = 0, max_value = 150, step = 1,
            value = 0,
            key = 'min_value',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        # Slider max_value recon error
        st.slider(
            label = "Value Max ",
            help = "All value above this will have the same color",
            min_value = 0, max_value = 150, step = 1,
            value = 55,
            key = 'max_value',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        # Slider fontsize recon error
        st.slider(
            label = "Fontsize (Error Map)",
            min_value = 5, max_value = 50, step = 1,
            value = 17,
            key = 'fontsize_error_map',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.slider(
            label = "Figure Width (Error Map)",
            help = "Height of the error map plot figure",
            min_value = 5, max_value = 40, step = 1,
            value = 15,
            key = 'figsize_width_error_map',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.slider(
            label = "Figure Height (Error Map)",
            help = "Height of the error map figure",
            min_value = 5, max_value = 40, step = 1,
            value = 15,
            key = 'figsize_height_error_map',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

    with option_col_2 :
        st.checkbox(
            label = 'Add runs grid',
            value = True,
            key = 'add_runs_grid',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.checkbox(
            label = 'Add Trials grid',
            value = False,
            key = 'add_trials_grid',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.checkbox(
            label = 'Add Colorbar',
            value = False,
            key = 'add_colorbar',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.checkbox(
            label = 'Transpose Plot',
            value = False,
            key = 'transpose_error_map',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        
        # Colormap selection.
        # When imported from matplotlib and sorted the map 56 is the Reds
        st.selectbox(
            label = 'Select the Colormap',
            options = support_variables.COLORMAPS, index = 56,
            key = 'colormap',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.checkbox(
            label = 'Highlight selected trial',
            help = 'Highlight the selected trial in the error map',
            value = False,
            key = 'visualize_trial_error_map',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.radio(
            label = "Highlight mode",
            help = "\"Circle\" add a circle around the selected trial in the error map. \"Cross\" draw two lines that intersect in the selected trial",
            options = ["Circle", "Cross", "Both"],
            key = "visualize_trial_error_map_mode",
            disabled = not st.session_state.visualize_trial_error_map,
            horizontal = True,
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

def build_trial_selection(streamlit_container_error_map_plot, streamlit_container_eeg_signal) :
    st.write("---")
    st.write("Trial Selection")

    col_1, col_2 = st.columns([0.5, 0.5], border = False)

    with col_1 :
        st.selectbox(
            label = 'Select the trial',
            options = np.arange(288) + 1, index = 0,
            key = 'trials',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

    with col_2 :
        st.selectbox(
            label = 'Select the Channel',
            options = support_variables.CHANNELS_D2A, index = 7,
            key = 'channel',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

def build_eeg_plot_options(streamlit_container_error_map_plot, streamlit_container_eeg_signal) :

    st.write("---")
    st.write("Options for the EEG Plot")

    option_col_1, option_col_2 = st.columns([0.5, 0.5], border = False, gap = "Large", vertical_alignment = "center")

    with option_col_1 :
        st.slider(
            label = "Fontsize (EEG Plot)",
            min_value = 5, max_value = 50, step = 1,
            value = 25,
            key = 'fontsize_eeg_plot',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.slider(
            label = "Figure Width (EEG Plot)",
            help = "Width of the EEG plot figure",
            min_value = 5, max_value = 40, step = 1,
            value = 15,
            key = 'figsize_width_eeg_plot',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.slider(
            label = "Figure Height (EEG Plot)",
            help = "Height of the EEG plot figure",
            min_value = 5, max_value = 40, step = 1,
            value = 7,
            key = 'figsize_height_eeg_plot',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

    with option_col_2 :

        st.checkbox(
            label = 'Show hvEEGNet reconstruction',
            help = 'Plot in red the EEG signal reconstructed through hvEEGNet',
            value = False,
            key = 'show_reconstruction',
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )

        st.radio(
            label = "EEG domain",
            help = "Decide if plot the EEG signal in time domain, frequency domain or both",
            options = ["Time", "Frequency", "Both"],
            key = "domain_eeg_signal",
            horizontal = True,
            on_change = support_plot.create_both_error_map_and_eeg_signal_plots,
            args = [streamlit_container_error_map_plot, streamlit_container_eeg_signal]
        )


