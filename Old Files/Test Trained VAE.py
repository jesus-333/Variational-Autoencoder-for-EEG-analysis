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
alternative='greater'

our_results = [84.84,75.23,90.15,75.27,82.38,84.76,87.23,88.98,91.77]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

wilcoxon_mon = stats.wilcoxon(our_results, mon_results, alternative = alternative)
wilcoxon_xu = stats.wilcoxon(our_results, xu_results, alternative = alternative)

t_test_mon = stats.ttest_ind(our_results, mon_results, equal_var = equal_var, alternative = alternative)
t_test_xu = stats.ttest_ind(our_results, xu_results, equal_var = equal_var, alternative = alternative)

ks_test_mon = stats.ks_2samp(our_results, mon_results, alternative = alternative)
ks_test_xu = stats.ks_2samp(our_results, xu_results, alternative = alternative)

print("Method: ", alternative, "(LOSS)")
print("P value (wilcoxon):\t{:.2f}\t{:.2f}".format(wilcoxon_mon[1] * 100, wilcoxon_xu[1] * 100))
print("P value (T-test):\t{:.2f}\t{:.2f}".format(t_test_mon[1] * 100, t_test_xu[1] * 100))
print("P value (KS-test):\t{:.2f}\t{:.2f}".format(ks_test_mon[1] * 100, ks_test_xu[1] * 100), "\n")

#%% END Results

import scipy.stats as stats
import numpy as np

equal_var = True
alternative='greater'


our_results = [83.56, 73.46, 89.39, 74.63, 81.4, 84.89, 86.82, 88.49, 91.81]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

# our_results = [0.781635917,	0.6358025,	0.857638833,	0.64988425,	0.740740667,	0.79359575,	0.832176,	0.837963,	0.886959833]
# mon_results = [0.67,	0.35,	0.65,	0.62,	0.58,	0.45,	0.69,	0.7,	0.64]
# xu_results = [0.8214,	0.4838,	0.7696,	0.6664,	0.5024,	0.5301,	0.7837,	0.8655,	0.8942]

if(our_results[0] > 2): data_type = 'Accuracy'
else: data_type = 'Kappa Score'

wilcoxon_mon = stats.wilcoxon(our_results, mon_results, alternative = alternative)
wilcoxon_xu = stats.wilcoxon(our_results, xu_results, alternative = alternative)

t_test_mon = stats.ttest_ind(our_results, mon_results, equal_var = equal_var, alternative = alternative)
t_test_xu = stats.ttest_ind(our_results, xu_results, equal_var = equal_var, alternative = alternative)

ks_test_mon = stats.ks_2samp(our_results, mon_results, alternative = alternative)
ks_test_xu = stats.ks_2samp(our_results, xu_results, alternative = alternative)

print("Method: ", alternative, "(END) - ", data_type)
# print("P value (wilcoxon):\t{:.2f}\t{:.2f}".format(wilcoxon_mon[1] * 100, wilcoxon_xu[1] * 100))
print("P value (T-test):\t{:.2f}\t{:.2f}".format(t_test_mon[1] * 100, t_test_xu[1] * 100))
# print("P value (KS-test):\t{:.2f}\t{:.2f}".format(ks_test_mon[1] * 100, ks_test_xu[1] * 100))

    

#%% Proportion Difference Test

from mlxtend.evaluate import proportion_difference

our_results = [83.56, 73.46, 89.39, 74.63, 81.4, 84.89, 86.82, 88.49, 91.81]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

p_value_mon = []
p_value_xu = []

for i in range(len(our_results)):
    acc_our = our_results[i] / 100
    acc_mon = mon_results[i] / 100
    acc_xu = xu_results[i] / 100
    
    z, p = proportion_difference(acc_our, acc_mon, n_1=288)
    p_value_mon.append(p)
    
    z, p = proportion_difference(acc_our, acc_xu, n_1=288)
    p_value_xu.append(p)
    
    
for i in range(9):
    print(i + 1, "\t", round(p_value_mon[i] * 100, 2), "\t", round(p_value_xu[i] * 100, 2))
    
print("AVG\t", round(proportion_difference(np.mean(our_results)/100, np.mean(mon_results)/100, n_1=288 * 9)[1] * 100, 2), "\t", round(proportion_difference(np.mean(our_results)/100, np.mean(xu_results)/100, n_1=288 * 9)[1] * 100, 2))

#%% Resampled Paired t-test

from scipy.stats import t as t_dist

def paired_t_test(p):
    p_hat = np.mean(p)
    n = len(p)
    den = np.sqrt(sum([(diff - p_hat)**2 for diff in p]) / (n - 1))
    t = (p_hat * (n**(1/2))) / den
    
    p_value = t_dist.sf(t, n-1)*2
    
    return t, p_value


our_results = [83.56, 73.46, 89.39, 74.63, 81.4, 84.89, 86.82, 88.49, 91.81]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

p_value_mon = []
p_value_xu = []

t, p = paired_t_test(np.asarray(our_results) - np.asarray(mon_results))
print("Monolith:\t", round(p * 100, 2))
t, p = paired_t_test(np.asarray(our_results) - np.asarray(xu_results))
print("Xu:\t\t\t", round(p * 100, 2))

#%% ANOVA

import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt

# Acccuracy
our_results = [83.56, 73.46, 89.39, 74.63, 81.4, 84.89, 86.82, 88.49, 91.81]
FBCSP_results = [76,	56.5,	81.25,	61,	55,	45.25,	82.75,	81.25,	70.75]
BO_results = [82.12,	44.86,	86.6,	66.28,	48.72,	53.3,	72.64,	82.33,	76.35]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
FBCSP_SVM_results = [82.29,	60.42,	82.99,	72.57,	60.07,	44.1,	86.11,	77.08,	75]
CW_CNN_results = [86.11,	60.76,	86.81,	67.36,	62.5,	45.14,	90.63,	81.25,	77.08]
SCSSP_results = [67.88,	42.18,	77.87,	51.77,	50.17,	45.97,	87.5,	85.79,	76.31]
DFNN_results = [83.2,	65.69,	90.29,	69.42,	61.65,	60.74,	85.18,	84.21,	85.48]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

# Kappa score
# our_results = [0.781635917,	0.6358025,	0.857638833,	0.64988425,	0.740740667,	0.79359575,	0.832176,	0.837963,	0.886959833]
# FBCSP_results = [0.86,	0.24,	0.7,	0.68,	0.36,	0.34,	0.66,	0.75,	0.82]
# BO_results = [0.6481,	0.3657,	0.6632,	0.5046,	0.3241,	0.2963,	0.7188,	0.6354,	0.6458]
# mon_results = [0.67,	0.35,	0.65,	0.62,	0.58,	0.45,	0.69,	0.7,	0.64]
# FBCSP_SVM_results = [0.764,	0.472,	0.773,	0.634,	0.468,	0.255,	0.815,	0.694,	0.667]
# CW_CNN_results = [0.815,	0.477,	0.824,	0.565,	0.5,	0.269,	0.875,	0.75,	0.694]
# SCSSP_results = [0.7407,	0.2685,	0.7685,	0.4259,	0.287,	0.2685,	0.7315,	0.7685,	0.7963]
# DFNN_results = [0.7,	0.32,	0.75,	0.54,	0.32,	0.34,	0.7,	0.69,	0.77]
# xu_results = [0.8214,	0.4838,	0.7696,	0.6664,	0.5024,	0.5301,	0.7837,	0.8655,	0.8942]


results_list = [our_results, FBCSP_results, BO_results, mon_results, FBCSP_SVM_results, CW_CNN_results, SCSSP_results, DFNN_results, xu_results]
std_list = []

for i in range(len(results_list)):
    # stats.probplot(results_list[i], dist="norm", plot=plt)
    # plt.title("Probability Plot - " +  str(i))
    # plt.show()
    
    std_list.append(np.std(results_list[i]))


print("Std ration = ", max(std_list) / min(std_list))

results_matrix = np.asarray(results_list).T


x_bar = np.mean(results_matrix)
SSTR = 9 * (np.mean(results_matrix, 0) - x_bar)**2
SS_between_group = np.sum(SSTR)

SSE = (9 - 1) * np.std(results_matrix, 0)**2
SS_withing_group = np.sum(SSE)

SSTR = np.sum(SSTR) + np.sum(SSE)

df_between_group = 9 - 1
df_within_group = 9 * len(results_list) - (9 - 1)
df_total = 9 * len(results_list) - 1

anova_table = [SS_between_group / df_between_group, SS_withing_group / df_within_group, SSTR / df_total]
print(anova_table)

# F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
# anova_table['F']['Between Groups'] = F
F_between_groups = anova_table[0] / anova_table[1]


# anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])
p_value = 1 - stats.f.cdf(F_between_groups,df_between_group, df_within_group)
print(p_value)

# F critical    
alpha = 0.1
# possible types "right-tailed, left-tailed, two-tailed"
tail_hypothesis_type = "two-tailed"
if tail_hypothesis_type == "two-tailed":
    alpha /= 2
    
# anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])
F_crit_within_group = stats.f.ppf(1-alpha, df_between_group, df_within_group)

# The p-value approach
print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
conclusion = "Failed to reject the null hypothesis."
if p_value <= alpha:
    conclusion = "Null Hypothesis is rejected."
print("F-score is:", F_crit_within_group, " and p value is:", p_value)    
print(conclusion)

#%% Chi square

std_mio = [1.501205594,	2.780002439,	1.474179107,	2.717169256,	2.452091364,	1.953223815, 1.511919644,	2.075995096,	1.164654963]
our_results = [83.56, 73.46, 89.39, 74.63, 81.4, 84.89, 86.82, 88.49, 91.81]
mon_results = [83.13,65.45,80.29,81.6,76.7,71.12,84,82.66,80.74]
xu_results = [86.6071,61.2613,87.2727,75.2,64.5455,65.9091,83.7838,89.9083,92.0792]

our_results = np.asarray(our_results)
mon_results = np.asarray(mon_results)
xu_results = np.asarray(xu_results)

t_our_mon = []
t_our_xu = []

# idx_del = [2, 5, 8]
# idx_del.reverse()

# for idx in idx_del:
#     del std_mio[idx]
#     del our_results[idx]
#     del mon_results[idx]
#     del xu_results[idx]

for i in range(len(xu_results)):
    num_1 = (our_results[i] - mon_results[i]) ** 2
    num_2 = (our_results[i] - xu_results[i]) ** 2
    
    den = std_mio[i] ** 2
    
    t_our_mon.append(num_1 / den)
    t_our_xu.append(num_2 / den)

print(t_our_mon)
print(t_our_xu)

print("Chi Square (mon):", np.sum(t_our_mon))
print("Chi Square (xu):", np.sum(t_our_xu))
