# -*- coding: utf-8 -*-
"""
File containing various support function.

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Import section

import numpy as np
import matplotlib.pyplot as plt
import os
import random

from scipy.io import loadmat, savemat
import scipy.signal as signal
import scipy.linalg as la
from scipy.fft import fft, fftfreq, rfft, rfftfreq

#%% Perturbation input functions (Dataset 2a)

def perturbateInputV1(trials_matrix, labels, false_freq_list, fs = 250, perturb_false_freq = False):
    
    perturbed_trials = np.zeros(trials_matrix.shape)
    
    for i in range(len(false_freq_list)):
        false_freq = false_freq_list[i]
        
        
        if(perturb_false_freq):
            # Each trials of a class is perturbed with a freq that +-0.2 the original false_freq
    
            for j in range(len(labels)):
                if(labels[j] == (i + 1)):
                     # Modify false freq
                     tmp_false_freq = false_freq + random.uniform(-0.2, 0.2)
                     
                     perturbation_signal = createPerturbationSignal(false_freq, trials_matrix.shape[2], amplitude = np.max(trials_matrix)/20, fs = fs)
                     perturbed_trials[j, :, :] = trials_matrix[j, :, :] + perturbation_signal
        else:
            # All the trials of a class are perturbed with the same false freq
            perturbation_signal = createPerturbationSignal(false_freq, trials_matrix.shape[2], amplitude = np.max(trials_matrix)/20, fs = fs)
            perturbed_trials[labels == (i + 1)] = trials_matrix[labels == (i + 1)] + perturbation_signal
        
    return perturbed_trials


def perturbateInputV2(trials_matrix, labels, false_freq_list, snr = 3, average_perturb = True, fs = 250):
    """
    Perturb the input trials matrix with sinusoidal signal of different frequency. 
    Each trial of a specifi class will be perturb by signal with a specifi frequency (e.g class 1 ---> 53Hz, class 2 ---> 89Hz etc)

    The signal to noise ration (snr) is the default set to 3dB.
    With average_perturb = True (defualt) the snr will be evaluated using the average power of the entire input matrix
    With average_perturb = False the snr will be evaluated trial per trial
    """
    
    # Matrix for the perturb signals
    # perturbed_trials = np.zeros(trials_matrix.shape)
    perturbed_trials = np.copy(trials_matrix)
    labels_value = np.unique(labels)
    
    for i in range(len(false_freq_list)):
        false_freq = false_freq_list[i]
        tmp_label = labels_value[i]
        
        if(average_perturb):
            # Find the trials associated with the current label
            tmp_index = labels == tmp_label
            
            # Evaluate the average power of the trials matrix
            average_power_trials_matrix = calculateTrialsAveragePower(trials_matrix)
            
            # Calculate the necessary input to obtain the snr required 
            tmp_amp = calculateAmplitudeFromSNRdB(snr, average_power_trials_matrix)
             
            # Create the noise signal and perturb the input
            perturbation_signal = createPerturbationSignal(false_freq, trials_matrix.shape[2], amplitude = tmp_amp, fs = fs)
            perturbed_trials[tmp_index] = trials_matrix[tmp_index] + perturbation_signal

        else:
            for j in range(len(labels)):
                if(labels[j] == tmp_label):
                    # Evaluate the average power for the trial
                    average_trial_power = calculateSingleTrialAveragePower(trials_matrix[j])
                    
                    # Calculate the necessary input to obtain the snr required 
                    tmp_amp = calculateAmplitudeFromSNRdB(snr, average_trial_power)
                    
                    # Create the noise signal and perturb the input
                    perturbation_signal = createPerturbationSignal(false_freq, trials_matrix.shape[2], amplitude = tmp_amp, fs = fs)
                    perturbed_trials[j] = trials_matrix[j] + perturbation_signal
                    
                    # print(j, average_trial_power, tmp_amp, computeSNR(trials_matrix[j, 0], perturbation_signal))
    
    return perturbed_trials
 

#%% Perturbation input function (HGD)

def perturbSingleChannelHGD(x, fs, snr = 3, perturb_freq = 50):
    average_ch_power = powerEvaluation(x)
            
    # Calculate the necessary input to obtain the snr required 
    tmp_amp = calculateAmplitudeFromSNRdB(snr, average_ch_power)
            
    # Create the noise signal and perturb the input
    perturbation_signal = createPerturbationSignal(perturb_freq, x.shape[0], amplitude = tmp_amp, fs = fs)
    # print(np.max(np.abs(x)), np.max(perturbation_signal))
    x = x + perturbation_signal
    
    return x
    

def perturbateHGDTrial(x, fs, snr = 3, perturb_freq = 50, channel_wise = False):
    perturb_x = np.zeros(x.shape)
    if(channel_wise):
        for i in range(x.shape[0]):
           perturb_x[i] = perturbSingleChannelHGD(x[i, :])
    else:
        # Compute average power of the trial
        average_trial_power = calculateSingleTrialAveragePower(x)
        
        # Calculate the necessary input to obtain the snr required 
        tmp_amp = calculateAmplitudeFromSNRdB(snr, average_trial_power)
        
        # Create the noise signal and perturb the input
        perturbation_signal = createPerturbationSignal(perturb_freq, x.shape[0], amplitude = tmp_amp, fs = fs)
        
        perturb_x = x + perturbation_signal


#%% Signal power related function

def powerEvaluation(x):
    """
    Calculate the power of signal x as the sum of abs(x_i)^2
    """
    
    tmp_sum = 0
    for i in range(len(x)): tmp_sum += abs(x[i])**2
    
    return (1/(2 * len(x))) * tmp_sum


def calculateSingleTrialAveragePower(trial):
    """
    Calculate the average power of a single trial of dimension "channel x samples"
    The average power is consider as the average power through channels
    """
    
    tmp_sum = 0
    for ch in trial: tmp_sum += powerEvaluation(ch)
    
    return tmp_sum/trial.shape[0]


def calculateTrialsAveragePower(trials):
    """
    Calculate the average power of n trials. The input dimension must be "n_trials x channel x samples"
    The average power of single trials is consider as the average power through channels.
    Once obtain the power for each trial the average is computed and returned.
    """
    
    tmp_sum = 0
    for trial in trials: tmp_sum += calculateSingleTrialAveragePower(trial)

    return tmp_sum/trials.shape[0]


def calculateAmplitudeFromSNRdB(snr_db, p_eeg):
    """
    Evaluate the amplitude of a sinusoidal signal to reach the given SNR in dB given the power p_eeg
    """
    
    # Evaluate noise power
    p_n = p_eeg * 10 ** (- snr_db /10)
    # p_n = p_eeg * (10 ** (snr_db /10)) ** -1
    
    # Evaluate amplitude
    amp = np.sqrt(2 * p_n)
    
    return amp


def computeSNR(x_signal, x_noise):
    p_signal = powerEvaluation(x_signal)
    p_noise = powerEvaluation(x_noise)

    return 10 * np.log10(p_signal/p_noise)    


def createPerturbationSignal(perturbation_freq, signal_length_in_samples, amplitude, fs = 250):
    time_duration = signal_length_in_samples / fs

    time_vector = np.linspace(0, time_duration, signal_length_in_samples)
    
    pertubation_signal = amplitude *  np.sin(2 * np.pi * perturbation_freq * time_vector)
    
    return pertubation_signal


#%% PSD related function

def computeSingleChannelPSD(x, fs = 250):
    freq, PSD_ch = signal.welch(x, fs = fs)
    return freq, PSD_ch 


def computeSingleTrialAveragePSD(trial, fs = 250, nperseg = 512):
    average_PSD = np.zeros(int(nperseg/2 + 1))
    
    for ch in trial:
        freq_tmp, data_tmp = signal.welch(ch, fs = fs, nperseg = nperseg)
        # print(average_PSD.shape, data_tmp.shape)
        average_PSD += data_tmp
        
    average_PSD = average_PSD / trial.shape[0]
    
    return freq_tmp, average_PSD


def computeDatasetAveragePSD_V1(dataset, idx_ch = 0, fs = 250):
    average_PSD = np.zeros(129)
    
    # Calculate PSD
    for el in dataset:
        ch = el[idx_ch]
        ch = np.squeeze(ch)
        freq_tmp, data_tmp = signal.welch(ch[:], fs = fs)
        
        average_PSD += data_tmp
    
    # Evalaute average PSD
    average_PSD = average_PSD / len(dataset)
    
    return freq_tmp, average_PSD


def computeDatasetAveragePSD_V2(dataset, idx_ch = 0, fs = 250):
    average_PSD = np.zeros(129)
    
    # Calculate PSD
    for trial in dataset:
        freq_tmp, data_tmp = computeSingleTrialAveragePSD(trial, fs = fs)
        average_PSD += data_tmp
    
    # Evalaute average PSD
    average_PSD = average_PSD / len(dataset)
    
    return freq_tmp, average_PSD

#%% FFT related function

def computeSingleChannelFFT(x, fs, use_right = True, rescale = True):
    if(use_right):
        fft_ch = rfft(x)
        freq = rfftfreq(len(x), 1/fs)
    else:
        fft_ch = fft(x)
        freq = fftfreq(len(x), 1/fs)
        
    if(rescale): fft_ch = fft_ch/len(x)
    
    return freq, fft_ch 


def computeSingleTrialAverageFFT(trial, fs = 250, nperseg = 512, use_right = True, rescale = True):
    # Create vector for store results
    freq_tmp, data_tmp = computeSingleChannelFFT(trial[0], fs, use_right = use_right, rescale = rescale)
    average_FFT = np.zeros(data_tmp.shape[0], dtype = 'complex128')
    
    # Evaluate average FFT
    for ch in trial:
        freq_tmp, data_tmp = computeSingleChannelFFT(ch, fs, use_right = use_right)
        average_FFT += abs(data_tmp)
        
    average_FFT = average_FFT / trial.shape[0]
    
    return freq_tmp, average_FFT


#%% Other functions

def filterSignal(data, fs, low_f, high_f, filter_order = 3):
    # Evaluate low buond and high bound in the [0, 1] range
    low_bound = low_f / (fs/2)
    high_bound = high_f / (fs/2)
    
    # Check input data
    if(low_bound < 0): low_bound = 0
    if(high_bound > 1): high_bound = 1
    if(low_bound > high_bound): low_bound, high_bound = high_bound, low_bound
    if(low_bound == high_bound): low_bound, high_bound = 0, 1
    
    b, a = signal.butter(filter_order, [low_bound, high_bound], 'bandpass')
    
    filtered_data = signal.filtfilt(b, a, data.T)
    return filtered_data.T

def cleanWorkspace():
    try:
        from IPython import get_ipython
        # get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass
