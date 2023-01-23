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

#%% END

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v23', type='model')
artifact_dir = artifact.download()