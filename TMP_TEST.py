#%% Imports

import sys
sys.path.insert(1, 'support')

import numpy as np
import wandb
import torch
import pandas as pd

import moabb_dataset as md
import config_file as cf
import metrics
import os

#%% Metrics download

version_list = [5,6,7,8,9]

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/Metrics:v9', type='metrics')
artifact_dir = artifact.download()

#%% Metrics reading

name_1 = 'metrics_BEST_CLF.csv'
name_2 = 'metrics_BEST_TOT.csv'
name_3 = 'metrics_END.csv'

name_list = [name_1, name_2, name_3]
best_sub = ''
best_version = ''
best_accuracy = 0

for version in version_list:
    
    for name in name_list:
        path = 'artifacts\Metrics-v{}\{}'.format(version, name)
        
        data = pd.read_csv(path)
        
        accuracy = data['accuracy'].to_numpy()
        kappa = data['cohen_kappa']
        
        if np.mean(accuracy) > best_accuracy:
            best_accuracy = np.mean(accuracy)
            best_sub = name
            best_version = version
            
    # print(version, best_accuracy)
    

path = 'artifacts\Metrics-v{}\{}'.format(best_version, best_sub)
best_accuracy = pd.read_csv(path)['accuracy'].to_numpy()
best_kappa = pd.read_csv(path)['cohen_kappa'].to_numpy()

print(best_sub)
print(best_version)
print(best_accuracy)
print(best_kappa)

#%% Download network 

# run = wandb.init()
# artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v55', type='model')
# artifact_dir = artifact.download()

hidden_space_dimension = artifact.metadata['train_config']['hidden_space_dimension']

tmp_path = './artifacts/vEEGNet_trained-v55'
network_weight_list = os.listdir(tmp_path)

#%% create Dataset (d2a old)

import support_datasets

subject_list = [1,2,3,4,5,6,7,8,9]

loader_list = []

for subj in subject_list:
    dataset_path = artifact.metadata['dataset_config']['test_path']
    dataset_path += '/{}/'.format(subj)
    test_data = support_datasets.PytorchDatasetEEGSingleSubject(dataset_path, normalize_trials = artifact.metadata['dataset_config']['normalize_trials'])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    loader_list.append(test_loader)
    
C = test_data[0][0].shape[1]
T = test_data[0][0].shape[2]

#%% create Dataset (moabb)

dataset_config = cf.get_moabb_dataset_config()
test_data = md.get_test_data(dataset_config)

C = test_data[0][0].shape[1]
T = test_data[0][0].shape[2]
print("C = {}\nT = {}".format(C, T))

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
loader_list = [test_loader]

#%%

import VAE_EEGNet

model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension, 
                                use_reparametrization_for_classification = False, 
                                print_var = False, tracking_input_dimension = False) 


metrics_per_file = metrics.compute_metrics_given_path(model, loader_list, 
                                                      tmp_path, device = 'cuda')

#%%

accuracy_list = []
for el in metrics_per_file: 
    tmp_metrics = np.asarray(el)
    accuracy_list.append(tmp_metrics[:, 0])

accuracy = np.asarray(accuracy_list).T

avg_accuracy_per_file = np.mean(accuracy, 0)
max_avg_accuracy = np.max(avg_accuracy_per_file)

print(max_avg_accuracy)

#%% PSD

import support_visualization as sv

version = 23
epoch_file = 'END'

# config = metrics.get_config_PSD()
config = dict(
    device = 'cuda', 
    fs = 128, # Origina sampling frequency
    window_length = 0.5 # Length of the window in second
)

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
model.load_state_dict(torch.load(path))

psd_original, psd_reconstructed, f = metrics.psd_reconstructed_output(model, test_data, 7, config)
print(psd_original.shape)

plot_config = dict(
    figsize = (15, 10),
    ch_list = ['C3', 'C4'],
    color_list = ['red', 'blue'],
    x_freq = f,
    font_size = 16,
)

sv.plot_psd_V1(psd_original, psd_reconstructed, plot_config)

#%% PSD (2)

import support_visualization as sv

version = 55
epoch_file = 'END'

# config = metrics.get_config_PSD()
config = dict(
    device = 'cuda', 
    fs = 128, # Origina sampling frequency
    window_length = 0.5 # Length of the window in second
)

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
model.load_state_dict(torch.load(path))

psd_original_1, psd_reconstructed_1, f = metrics.psd_reconstructed_output(model, test_data, 0, config)
psd_original_2, psd_reconstructed_2, f = metrics.psd_reconstructed_output(model, test_data, 127, config)

psd_original_list = [psd_original_1, psd_original_2]
psd_reconstructed_list = [psd_reconstructed_1, psd_reconstructed_2]

plot_config = dict(
    figsize = (15, 10),
    ch_list = ['C3', 'C4'],
    color_list = ['red', 'blue'],
    x_freq = f,
    font_size = 16,
)

# sv.plot_psd_V2(psd_original_list, psd_reconstructed_list, plot_config)

import matplotlib.pyplot as plt
channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
channel_list = np.asarray(channel_list)
plt.figure(figsize = (15, 10))

idx_ch = channel_list == 'C3'
plt.plot(f, psd_reconstructed_1[idx_ch].squeeze(), label = 'C3 - RIGHT')
plt.plot(f, psd_reconstructed_2[idx_ch].squeeze(), label = 'C3 - LEFT')

idx_ch = channel_list == 'C4'
plt.plot(f, psd_reconstructed_1[idx_ch].squeeze(), label = 'C4 - RIGHT')
plt.plot(f, psd_reconstructed_2[idx_ch].squeeze(), label = 'C4 - LEFT')

plt.legend()
plt.tight_layout()

#%%

for i in range(1):

    sample = test_data[i][0].unsqueeze(0)
    label =  test_data[i][1].unsqueeze(0)
    if label == 0: label = 'LEFT'
    if label == 1: label = 'RIGHT'
    if label == 2: label = 'FOOT'
    if label == 3: label = 'TONGUE'
    
    x_r_mean, x_r_std, mu, log_var, _ = model(sample)
    
    x = sample.cpu().squeeze().detach().numpy()
    x_r = x_r_mean.cpu().squeeze().detach().numpy()
    
    f = np.linspace(0, 4, x.shape[1])
    plt.figure(figsize = (15, 10))
    
    idx_ch = channel_list == 'C3'
    plt.plot(f, x_r[0], label = 'Reconstructed')
    plt.title(label)
    
    idx_ch = channel_list == 'C4'
    plt.plot(f, x[0].squeeze(), label = 'Original')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


#%% END

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v23', type='model')
artifact_dir = artifact.download()