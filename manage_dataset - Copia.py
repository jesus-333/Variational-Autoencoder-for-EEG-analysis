# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%%
import sys
sys.path.insert(1, 'support')

from support.support_function_HGD import downloadDataset, computeTrialsHGD, trialLengthDensity
from support.support_perturbation import perturbSingleChannelHGDWithImpulse, perturbSingleChannelHGDWithGaussianNoise, perturbateHGDTrial
from support.support_perturbation import computeSingleChannelPSD, computeSingleTrialAveragePSD, computeDatasetAveragePSD_V2
from support.support_perturbation import computeSingleChannelFFT, computeSingleTrialAverageFFT, filterSignal

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#%%

name_dataset = 'HGD'
type_dataset = 'Train'
idx_subject = [3]

download_dataset = False
resampling = True

path_dataset = 'Dataset/' + name_dataset + '/' + type_dataset + '/'

#%% Download

if(download_dataset):
    for idx in idx_subject: downloadDataset(idx, path = 'Dataset/{}/'.format(name_dataset))
    
#%% Division in trials
for idx in idx_subject: computeTrialsHGD(idx = idx, path_dataset = path_dataset, resampling = resampling)

#%% Pertrubation (TMP)(IMPULSE NOISE)
idx_subj = 3
idx_ch = 22
n = 33
fs = 500
snr = 3
perturb_freq = 50
low_f_filter = 0.5
high_f_filter = 249
time_section = [0, 3]
freq_section = [0, 125]
figsize = (20, 15)

name_dataset = 'HGD'
type_dataset = 'Train'

use_right = True
rescale = True
use_log_scale = False
filter_signal = True
remove_zero_sample_FFT = False

# Path for the file
path_dataset = 'Dataset/' + name_dataset + '/' + type_dataset + '/' + str(idx_subj) + '/' + str(n)

# Load trial
x = loadmat(path_dataset)['trial']

# Extract channel and perturb
x_ch = x[idx_ch]
if(filter_signal): x_ch = filterSignal(x_ch, fs, low_f_filter, high_f_filter)
x_ch_perturb = perturbSingleChannelHGDWithImpulse(x_ch, fs, snr = snr, perturb_freq = perturb_freq)
t_vet = np.linspace(0, len(x_ch)/fs, len(x_ch))

# Compute FFT
freq_tmp, fft_ch = computeSingleChannelFFT(x_ch, fs, use_right = use_right, rescale = rescale)
freq_tmp, fft_ch_perturb = computeSingleChannelFFT(x_ch_perturb, fs, use_right = use_right, rescale = rescale)

if(remove_zero_sample_FFT):
    freq_tmp = freq_tmp[1:]
    fft_ch = fft_ch[1:]
    fft_ch_perturb = fft_ch_perturb[1:]

# Plot related command
fig, axs = plt.subplots(2, 2, figsize = figsize)

axs[0,0].plot(t_vet, x_ch)
axs[0,0].set_title('Signal in Time (RAW)')
axs[0,0].set_xlabel('Time [s]')
axs[0,0].set_xlim(time_section)

axs[0,1].plot(freq_tmp, abs(fft_ch))
axs[0,1].set_title('Signal in Frequency (RAW)')
axs[0,1].set_xlabel('Frequency [Hz]')
axs[0,1].set_xlim(freq_section)
if(use_log_scale): axs[0,1].set_xscale('log')

axs[1,0].plot(t_vet, x_ch_perturb)
axs[1,0].set_title('Signal in Time (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[1,0].set_xlabel('Time [s]')
axs[1,0].set_xlim(time_section)

axs[1,1].plot(freq_tmp, abs(fft_ch_perturb))
axs[1,1].set_title('Signal in Frequency (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[1,1].set_xlabel('Frequency [Hz]')
axs[1,1].set_xlim(freq_section)
if(use_log_scale): axs[1,1].set_xscale('log')



#%% Pertrubation (TMP)(GAUSSIAN NOISE)
idx_subj = 3
idx_ch = 22
n = 33
fs = 500
snr = 6
perturb_freq = 50
low_f_filter = 0.5
high_f_filter = 249
low_f_noise = 70
high_f_noise = 90
time_section = [0, 3]
freq_section = [0, 125]
figsize = (20, 15)

name_dataset = 'HGD'
type_dataset = 'Train'

use_right = True
rescale = True
use_log_scale = False
filter_signal = True

# Path for the file
path_dataset = 'Dataset/' + name_dataset + '/' + type_dataset + '/' + str(idx_subj) + '/' + str(n)

# Load trial
x = loadmat(path_dataset)['trial']
t_vet = np.linspace(0, len(x_ch)/fs, len(x_ch))

# Extract channel and perturb
x_ch = x[idx_ch]
if(filter_signal): x_ch = filterSignal(x_ch, fs, low_f_filter, high_f_filter)

x_1, x_2 = perturbSingleChannelHGDWithGaussianNoise(x_ch, fs, snr = snr, low_f = low_f_noise, high_f= high_f_noise)

# Compute FFT
freq_tmp, fft_ch = computeSingleChannelFFT(x_ch, fs, use_right = use_right, rescale = rescale)
freq_tmp, fft_ch_perturb_1 = computeSingleChannelFFT(x_1, fs, use_right = use_right, rescale = rescale)
freq_tmp, fft_ch_perturb_2 = computeSingleChannelFFT(x_2, fs, use_right = use_right, rescale = rescale)

# Plot related command
fig, axs = plt.subplots(3, 2, figsize = figsize)

axs[0,0].plot(t_vet, x_ch)
axs[0,0].set_title('Signal in Time (RAW)')
axs[0,0].set_xlabel('Time [s]')
axs[0,0].set_xlim(time_section)

axs[0,1].plot(freq_tmp, abs(fft_ch))
axs[0,1].set_title('Signal in Frequency (RAW)')
axs[0,1].set_xlabel('Frequency [Hz]')
axs[0,1].set_xlim(freq_section)
if(use_log_scale): axs[0,1].set_xscale('log')

axs[1,0].plot(t_vet, x_1)
axs[1,0].set_title('Signal in Time (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[1,0].set_xlabel('Time [s]')
axs[1,0].set_xlim(time_section)

axs[1,1].plot(freq_tmp, abs(fft_ch_perturb_1))
axs[1,1].set_title('Signal in Frequency (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[1,1].set_xlabel('Frequency [Hz]')
axs[1,1].set_xlim(freq_section)
if(use_log_scale): axs[1,1].set_xscale('log')

axs[2,0].plot(t_vet, x_2)
axs[2,0].set_title('Signal in Time (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[2,0].set_xlabel('Time [s]')
axs[2,0].set_xlim(time_section)

axs[2,1].plot(freq_tmp, abs(fft_ch_perturb_2))
axs[2,1].set_title('Signal in Frequency (PERTURB)(f = {}Hz, SNR = {}dB)'.format(perturb_freq, snr))
axs[2,1].set_xlabel('Frequency [Hz]')
axs[2,1].set_xlim(freq_section)
if(use_log_scale): axs[1,1].set_xscale('log')


    