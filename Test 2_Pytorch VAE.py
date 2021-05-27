"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%% Path for imports

import sys
sys.path.insert(1, 'support')

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch

from support.DynamicNet import DynamicNet
from support.support_EEGNet import getParametersEncoder, getParametersDecoder
from support.VAE_EEGNet import EEGNetVAE

#%% Load data (TRIAL)

hidden_space_dimension = 4

idx_subj = 3
idx_ch = 22
n_trial = 33

name_dataset = 'HGD'
type_dataset = 'Train'

# Path for the file
path_dataset = 'Dataset/' + name_dataset + '/' + type_dataset + '/' + str(idx_subj) + '/' + str(n_trial)

# Load trial
x = loadmat(path_dataset)['trial']
x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()

C = x.shape[2]
T = x.shape[3]

#%% Encoder (creation and test)

parameters_encoder = getParametersEncoder(C = x.shape[2], T = x.shape[3])

encoder = DynamicNet(parameters_encoder, print_var = False, tracking_input_dimension = False)

z = encoder(x)

print("x.shape = ", x.shape)
print("z.shape = ", z.shape)

#%% Complete VAE (creation and test)

vae = EEGNetVAE(C = C, T = T, hidden_space_dimension = hidden_space_dimension, tracking_input_dimension = True)


x_r = vae(x)

# print(x_r.shape)


