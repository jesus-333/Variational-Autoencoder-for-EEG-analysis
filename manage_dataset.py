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

#%% Train dataset

idx_subj = 3
n = 33
fs = 500
snr = 6
perturb_freq = 50
low_f_filter = 0.5
high_f_filter = 249
low_f_noise = 70
high_f_noise = 90

name_dataset = 'HGD'
type_dataset = 'Train'

use_right = True
rescale = True
use_log_scale = False
filter_signal = True

# Path for the dataset
path_dataset = 'Dataset/' + name_dataset + '/' + type_dataset + '/' + str(idx_subj) + '/' + str(n)

