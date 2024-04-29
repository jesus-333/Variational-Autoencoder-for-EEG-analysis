"""
This files contain various functions to normalize EEG data

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def z_score(x_eeg, x_mean : float = None, x_std : float = None) :
    """
    Normalize the data in x_eeg through z_score (i.e. remove mean and divide by standard deviation).
    Note that in this functions a single mean and a single standard deviation are used during normalization, i.e. I remove from all the data the same mean and divided by the same std.
    If you want to normalize trial by trial (or channel by channel) check the function z_score_trial_by_trial, z_score_trial_by_trial_ch_by_ch and z_score_ch_by_ch

    @param x_eeg  : Tensor or numpy array of shape B x C x T with B = batch size, C = Channel size, T = N. of time samples
    @param x_mean : (float) Mean to use during normalization. By default is None and it is computed from x_eeg.
    @param x_std  : (float) Standard deviation to use during normalization. By default is None and it is computed from x_eeg.
    @return x_eeg_norm : Normalized EEG data.
    """
    
    # Check mean and std
    if x_mean is None : x_mean = np.mean(x_eeg, axis = (1, 2))
    if x_std is None  : x_std = np.mean(x_eeg, axis = (1, 2))

    # Normalize data
    x_eeg_norm = (x_eeg - x_mean) / x_std
    
    return x_eeg_norm

def z_score_trial_by_trial(x_eeg) :
    pass

def z_score_trial_by_trial_ch_by_ch(x_eeg) :
    pass

def z_score_ch_by_ch(x_eeg) :
    pass
