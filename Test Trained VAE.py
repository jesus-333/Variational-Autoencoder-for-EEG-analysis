# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%%

import sys
sys.path.insert(1, 'support')

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

from support.support_datasets import PytorchDatasetEEGSingleSubject, PytorchDatasetEEGMergeSubject
from support.VAE_EEGNet import EEGFramework

#%% Variables

hidden_space_dimension = 64
batch_size = 15

print_var = True
tracking_input_dimension = True
normalize_trials = True

merge_list = [1,2,3,4,5,6,7,8,9]

weights_file_name = 'eeg_framework_normal_loss_499.pth'

#%% Create model and load datasets

# Training dataset
path = 'Dataset/D2A/v2_raw_128/Train/'
train_dataset = PytorchDatasetEEGMergeSubject(path, idx_list = merge_list, normalize_trials = normalize_trials)
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

# Test dataset
path = 'Dataset/D2A/v2_raw_128/Test/'
test_dataset = PytorchDatasetEEGMergeSubject(path, idx_list = merge_list, normalize_trials = normalize_trials)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

# Create model
C = train_dataset[0][0].shape[1]
T = train_dataset[0][0].shape[2]
eeg_framework = EEGFramework(C = C, T = T, hidden_space_dimension = hidden_space_dimension, print_var = print_var, tracking_input_dimension = tracking_input_dimension)


#%% Load pretrained weights

path_weights = 'Saved model/Script 2 (Normal loss)(64 Hidden space)(D2A)(3)/'+ weights_file_name
weights = torch.load(path_weights)

new_weights = OrderedDict()
key_to_remove = []


# Step necessary because the in the saved state dict the classifiers key have different name respect the ones of the new model created
# E.g. : MODEL CREATED KEY: classifier.classifier.0.weight      SAVED KEY: classifier.0.weight
# So I created a new orderd dict with where the name of the classifier's layers are corrected and the rest are simply copied
for layer_name in weights:
    if('classifier' in layer_name): 
        new_layer_name = 'classifier.' + layer_name
        new_weights[new_layer_name] = weights[layer_name]
        key_to_remove.append(layer_name)
    else:
        new_weights[layer_name] = weights[layer_name]
        
eeg_framework.load_state_dict(new_weights)
eeg_framework.eval()

#%%

idx = 599

x = train_dataset[idx][0]

x_r = eeg_framework(x.unsqueeze(0))[0]

x = x.squeeze().detach().numpy()
x_r = x_r.squeeze()
x_r = (x_r - torch.min(x_r)) / (torch.max(x_r) - torch.min(x_r))
x_r = x_r.detach().numpy()

x_gen = np.random.normal(x_r)

for i in range(x.shape[0]):
    # fig, axs = plt.subplots(2, figsize = (15, 10))
    # axs[0].plot(x[i])
    # axs[1].plot(x_r[i])
    
    plt.figure(figsize = (20, 10))
    plt.plot(x[i])
    plt.plot(x_r[i])
    plt.plot(x_gen[i])
    plt.show()
    
    
#%% 

generator = eeg_framework.vae.decoder
generator.eval().float()

z = torch.normal(0, 1, size = (1, hidden_space_dimension))
x_r = generator(z).squeeze()

for i in range(x_r.shape[0]):
    plt.figure(figsize = (15, 10))
    plt.plot(x_r[i].detach().numpy())
    plt.show()
    
    
#%%

import scipy.stats as stats
import numpy as np

equal_var = True
alternative='two-sided'

our_results = [84.84,75.23,90.15,75.27,82.38,84.76,87.23,88.98,91.77]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

wilcoxon_mon = stats.wilcoxon(our_results, mon_results, alternative = alternative)
wilcoxon_xu = stats.wilcoxon(our_results, xu_results, alternative = alternative)

t_test_mon = stats.ttest_ind(our_results, mon_results, equal_var = equal_var, alternative = alternative)
t_test_xu = stats.ttest_ind(our_results, xu_results, equal_var = equal_var, alternative = alternative)

ks_test_mon = stats.ks_2samp(our_results, mon_results, alternative = alternative)
ks_test_xu = stats.ks_2samp(our_results, xu_results, alternative = alternative)

print("Method: ", alternative)
print("P value (wilcoxon):\t{:.2f}\t{:.2f}".format(wilcoxon_mon[1] * 100, wilcoxon_xu[1] * 100))
print("P value (T-test):\t{:.2f}\t{:.2f}".format(t_test_mon[1] * 100, t_test_xu[1] * 100))
print("P value (KS-test):\t{:.2f}\t{:.2f}".format(ks_test_mon[1] * 100, ks_test_xu[1] * 100))

#%%

import pyperclip as pc
import numpy as np
import scipy.stats as stats

txt = pc.paste()

a = np.zeros((9, 8))
i, j = 0, 0

for line in txt.split("\n"):
    j = 0
    for number in line.split("\t"):
        a[i, j] = float(number)
        j += 1
    i += 1

wilcoxon_pvalue = np.zeros((1, 8))
ttest_pvalue = np.zeros((1, 8))
our_results = [84.84,75.23,90.15,75.27,82.38,84.76,87.23,88.98,91.77]
# our_results = [0.80,0.68,0.87,0.66,0.76,0.80,0.83,0.85,0.89]
for i in range(8):
    wilcoxon_pvalue[0, i] = stats.wilcoxon(our_results, a[:, i], alternative = 'greater')[1]
    ttest_pvalue[0, i] = stats.ttest_ind(our_results, a[:, i], equal_var = True, alternative = 'greater')[1]
    
for i in range(len(ttest_pvalue[0])):
    print(np.round(ttest_pvalue[0, i] * 100, 2), "\t", np.round(wilcoxon_pvalue[0, i] * 100, 2))