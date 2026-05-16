"""
Contains functions used to load data

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import scipy.signal as signal
import streamlit as st


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

@st.cache_data
def load_recon_error_file(path_file : str) :
    """
    Load the file with the reconstruction error.
    Even if it is a single instruction, written as a functions allow me to use the cache_data decorator provided by streamlit
    """

    return np.load(path_file)

@st.cache_data
def load_eeg_data_file(path_recon_error_file : str) :
    """
    Load eeg data file.
    Even if it is a single instruction, written as a functions allow me to use the cache_data decorator provided by streamlit
    """

    return np.load(path_recon_error_file)


def compute_spectra(x_eeg_time : np.ndarray, nperseg : int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spectra of the eeg data with Welch method.
    """

    f, x_eeg_freq = signal.welch(x_eeg_time, fs = 250, nperseg = nperseg)
    
    return f, x_eeg_freq
