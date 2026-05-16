"""
Main script for the streamlit interface

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import streamlit as st

import support_interace
import support_plot

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

debug = True
column_gap = 'medium'
column_proportions = [1, 1]

st.set_page_config(
    page_title = "hvEEGNet Anomaly Detection",
    layout="wide"
)

st.header("hvEEGNet Anomaly Detection", divider = True)


plot_option, error_map_plot_space, eeg_signal_section = st.columns([0.2, 0.4, 0.4], border = debug, gap = column_gap)


with error_map_plot_space :
    st.subheader(
        body = "Error Map plot",
        divider = True
    )

with eeg_signal_section :
    st.subheader(
        body = "EEG Signal Visualization",
        divider = True
    )
    
with plot_option :
    st.subheader(
        body = "Settings",
        divider = True
    )

    support_interace.build_subject_selection(error_map_plot_space, eeg_signal_section)

    support_interace.build_recon_error_options(error_map_plot_space, eeg_signal_section)

    support_interace.build_trial_selection(error_map_plot_space, eeg_signal_section)

    support_interace.build_eeg_plot_options(error_map_plot_space, eeg_signal_section)


# support_plot.create_recon_error_map_train_and_test(error_map_plot_space)
