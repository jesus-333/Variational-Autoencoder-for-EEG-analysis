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
    if x_mean is None : x_mean = np.mean(x_eeg)
    if x_std is None  : x_std  = np.std(x_eeg)

    # Normalize data
    x_eeg_norm = (x_eeg - x_mean) / x_std
    
    return x_eeg_norm

def z_score_trial_by_trial(x_eeg) :
    """
    Normalize EEG data. Each trial is normalized with its mean and its standard deviation.
    Note that all channels are considered together, i.e. from a matrix of shape C x T a single mean and a single std are computed.
    In this way the trial, considered as a matrix of shape C x T will have 0 mean and 1 std

    @param x_eeg  : Tensor or numpy array of shape B x C x T with B = batch size, C = Channel size, T = N. of time samples
    @return x_eeg_norm : Normalized EEG data.
    """
 
    # TODO Vectorize
    x_eeg_norm = np.zeros(x_eeg.shape)
    for i in range(x_eeg.shape[0]) :
        x_trial = x_eeg[i]
        
        x_eeg_norm[i] = (x_trial - x_trial.mean()) / x_trial.std()

    return x_eeg_norm

def z_score_trial_by_trial_ch_by_ch(x_eeg) :
    """
    Normalize EEG data. Each channel of each trial is normalized with its mean and its standard deviation.
    Note that in this case each channel will have 0 mean and 1 std after the normalization.

    @param x_eeg  : Tensor or numpy array of shape B x C x T with B = batch size, C = Channel size, T = N. of time samples
    @return x_eeg_norm : Normalized EEG data.
    """
    
    # TODO Vectorize
    x_eeg_norm = np.zeros(x_eeg.shape)
    for i in range(x_eeg.shape[0]) :
        for j in range(x_eeg.shape[1]) :
            x_ch = x_eeg[i, j]
            
            x_eeg_norm[i, j] = (x_ch - x_ch.mean()) / x_ch.std()

    return x_eeg_norm

def z_score_ch_by_ch(x_eeg, x_mean : float = None, x_std : float = None) :
    """
    Normalize EEG data. Each channel of each trial is normalized with its mean and its standard deviation.
    Note that in this case each channel will have 0 mean and 1 std after the normalization.

    @param x_eeg  : Tensor or numpy array of shape B x C x T with B = batch size, C = Channel size, T = N. of time samples
    @return x_eeg_norm : Normalized EEG data.
    """

    # Check mean and std
    x_mean = np.mean(x_eeg, axis = (0, 2))
    x_std = np.std(x_eeg, axis = (0, 2))


    # Normalize data
    x_eeg_norm = (x_eeg - x_mean[None, :, None]) / x_std[None, :, None]

    return x_eeg_norm


def tmp_norm(x_eeg)  :
    x_eeg_norm = np.zeros(x_eeg.shape)
    for i in range(x_eeg.shape[0]) :
        for j in range(x_eeg.shape[1]) :
            x_ch = x_eeg[i, j]
            
            x_eeg_norm[i, j] = (x_ch - x_ch.mean()) / x_ch.std()

    return x_eeg_norm
