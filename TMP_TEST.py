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
import VAE_EEGNet
import matplotlib
import matplotlib.pyplot as plt

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

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v21', type='model')
artifact_dir = artifact.download()

hidden_space_dimension = artifact.metadata['train_config']['hidden_space_dimension']

tmp_path = './artifacts/vEEGNet_trained-v55'
network_weight_list = os.listdir(tmp_path)

#%% create Dataset (d2a old)

import support_datasets

subject_list = [1,2,3,4,5,6,7,8,9]

loader_list = []

# for subj in subject_list:
#     # dataset_path = artifact.metadata['dataset_config']['test_path']
#     dataset_path = 'Dataset/D2A/v2_raw_128/Test/{}/'.format(subj)
#     test_data = support_datasets.PytorchDatasetEEGSingleSubject(dataset_path, normalize_trials = False)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
#     dataset_path = 'Dataset/D2A/v2_raw_128/Train/{}/'.format(subj)
#     train_data = support_datasets.PytorchDatasetEEGSingleSubject(dataset_path, normalize_trials = False)
#     train_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
#     loader_list.append(test_loader)


dataset_path = 'Dataset/D2A/v2_raw_128/Test/'
test_data = support_datasets.PytorchDatasetEEGMergeSubject(dataset_path, subject_list, normalize_trials = False, optimize_memory = False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)

dataset_path = 'Dataset/D2A/v2_raw_128/Train/'
train_data = support_datasets.PytorchDatasetEEGMergeSubject(dataset_path, subject_list, normalize_trials = False, optimize_memory = False)
train_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
C = test_data[0][0].shape[1]
T = test_data[0][0].shape[2]

#%% create Dataset (moabb)

dataset_config = cf.get_moabb_dataset_config()
# dataset_config['resample_freq'] = 149
train_data = md.get_train_data(dataset_config)
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

psd_original_1, psd_reconstructed_1, f = metrics.psd_reconstructed_output(model, test_data, 7, config)
psd_original_2, psd_reconstructed_2, f = metrics.psd_reconstructed_output(model, test_data, 0, config)
psd_original_3, psd_reconstructed_3, f = metrics.psd_reconstructed_output(model, test_data, 433, config)
psd_original_4, psd_reconstructed_4, f = metrics.psd_reconstructed_output(model, test_data, 999, config)


psd_original_list = [psd_original_1, psd_original_2, psd_original_3]
psd_reconstructed_list = [psd_reconstructed_1, psd_reconstructed_2, psd_reconstructed_3]

plot_config = dict(
    figsize = (15, 10),
    ch_list = ['C3', 'C4'],
    color_list = ['red', 'blue', 'green'],
    x_freq = f,
    font_size = 16,
)

# sv.plot_psd_V2(psd_original_list, psd_reconstructed_list, plot_config)

#%% PLORT X_R

import matplotlib
import matplotlib.pyplot as plt

channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
channel_list = np.asarray(channel_list)
idx_C3 = channel_list == 'C3'
idx_C4 = channel_list == 'C4'
idx_CZ = channel_list == 'Cz'

idx = 1
left_list = [0]
right_list = [1]

sample = test_data[idx][0].unsqueeze(0)
label =  test_data[idx][1].unsqueeze(0)
if label == 0: label = 'LEFT'
if label == 1: label = 'RIGHT'
if label == 2: label = 'FOOT'
if label == 3: label = 'TONGUE'
print(label)

x_r_mean, x_r_std, mu, log_var, _ = model(sample.cuda())

x = sample.cpu().squeeze().detach().numpy()
x_r = x_r_mean.cpu().squeeze().detach().numpy()


plt.figure(figsize = (15,10))
t = np.linspace(0, 4, T)

plt.plot(t, x[idx_CZ].squeeze(), linestyle = 'solid' , 
         label = 'Original signal', linewidth = 2, color = 'grey', alpha = 0.55)
plt.plot(t, x_r[idx_CZ].squeeze(), linestyle = 'solid' ,  
         label = 'Reconstructed Signal', linewidth = 2, color = 'black')


legend_properties = {'weight':'bold'}
plt.legend()

plt.xlabel("Time [s]", fontweight='bold')
plt.ylabel(r'Amplitude [$\mathbf{\mu V}$]', fontweight='bold')
plt.xlim([0, 4])
plt.ylim([-20, 20])
plt.rcParams.update({'font.size': 18})
plt.rcParams["font.weight"] = "bold"
plt.tight_layout()
plt.grid(True)

name = 'original vs reconstructed_CZ_single trial'

file_type = 'png'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

file_type = 'eps'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

#%% PLOT PSD 

import matplotlib
import matplotlib.pyplot as plt

channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
channel_list = np.asarray(channel_list)
idx_C3 = channel_list == 'C3'
idx_C4 = channel_list == 'C4'
idx_CZ = channel_list == 'Cz'

idx_foot_1 = channel_list == 'FC3'
idx_foot_2 = channel_list == 'FC4'

psd = psd_reconstructed_1
plt.figure(figsize = (15,10))

plt.plot(f, psd[idx_C3].squeeze(), linestyle = 'solid' , 
         label = 'C3', linewidth = 2, color = 'red')
plt.plot(f, psd[idx_C4].squeeze(), linestyle = 'dashed' ,  
         label = 'C4', linewidth = 2, color = 'blue')
plt.plot(f, psd[idx_CZ].squeeze(), linestyle = 'dashdot' ,  
         label = 'CZ', linewidth = 2, color = 'green')
plt.plot(f, ((psd[idx_foot_1] + psd[idx_foot_1]/2).squeeze()), linestyle = 'dotted' ,  
         label = '(FC3 + FC4)/2', linewidth = 2, color = 'black')

legend_properties = {'weight':'bold'}
plt.legend()

plt.xlabel("Frequency [Hz]", fontweight='bold')
plt.ylabel(r'PSD [$\mathbf{\mu V^2}$/Hz]', fontweight='bold')
plt.rcParams.update({'font.size': 22})
plt.ylim([0, 0.37])
plt.xlim([0, 15])
# plt.yscale('log')
plt.rcParams["font.weight"] = "bold"
plt.tight_layout()
plt.grid(True)

name = 'RH_15'

file_type = 'png'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

file_type = 'eps'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

#%%

version = 21
epoch_file = 'END'

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
state_dict = torch.load(path)

hidden_space_dimension = int(state_dict['vae.encoder.fc_encoder.bias'].shape[0] / 2) 

model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension, 
                                use_reparametrization_for_classification = False, 
                                print_var = False, tracking_input_dimension = False) 


model.load_state_dict(state_dict)

left_list = []
right_list = []
foot_list = []
tongue_list = []

for i in range(1000):

    sample = test_data[i][0].unsqueeze(0)
    label =  test_data[i][1].unsqueeze(0)
    if label == 0: label = 'LEFT'
    if label == 1: label = 'RIGHT'
    if label == 2: label = 'FOOT'
    if label == 3: label = 'TONGUE'
    
    if label == 'LEFT': left_list.append(i)
    if label == 'RIGHT': right_list.append(i)
    if label == 'FOOT': foot_list.append(i)
    if label == 'TONGUE': tongue_list.append(i)
    
model.eval()
tmp_list = left_list
channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
channel_list = np.asarray(channel_list)
idx_C3 = channel_list == 'C3'
idx_C4 = channel_list == 'C4'
idx_CZ = channel_list == 'Cz'

idx_foot_1 = channel_list == 'FC3'
idx_foot_2 = channel_list == 'FC4'

n_el = 50

avg_c3 = np.zeros(T)
avg_c4 = np.zeros(T)
avg_cz = np.zeros(T)
avg_tongue = np.zeros(T)

model.cuda()

for i in range(n_el):
    sample = test_data[tmp_list[i]][0].unsqueeze(0)
    label =  test_data[tmp_list[i]][1].unsqueeze(0)

    x_r_mean, x_r_std, mu, log_var, _ = model(sample.cuda())
    
    x   = sample.cpu().squeeze().detach().numpy()
    x_r = x_r_mean.cpu().squeeze().detach().numpy()

    avg_c3 += x_r[idx_C3].squeeze()
    avg_c4 += x_r[idx_C4].squeeze()
    avg_cz += x_r[idx_CZ].squeeze()
    avg_tongue += (x_r[idx_foot_1] + x_r[idx_foot_1]/2).squeeze()
    
avg_c3 /= n_el
avg_c4 /= n_el
avg_cz /= n_el
avg_tongue /= n_el

print(np.mean(avg_c3), np.mean(avg_c4), np.mean(avg_cz))

plt.figure(figsize = (15,10))
t = np.linspace(0, 4, T)

plt.plot(t, avg_c3, linestyle = 'solid' , 
         label = 'C3', linewidth = 2, color = 'red', )
plt.plot(t, avg_c4, linestyle = 'dashed' ,  
         label = 'C4', linewidth = 2, color = 'blue')
plt.plot(t, avg_cz, linestyle = 'dashdot' ,  
         label = 'CZ', linewidth = 2, color = 'green')
plt.plot(t, avg_tongue, linestyle = 'dotted' ,  
         label = '(FC3 + FC4)/2', linewidth = 2, color = 'black')

plt.legend()
plt.xlabel("Time [s]", fontweight='bold')
plt.ylabel(r'Amplitude [$\mathbf{\mu V^2}$]', fontweight='bold')
plt.xlim([0, 4])
plt.rcParams.update({'font.size': 22})
plt.tight_layout()
plt.grid(True)

name = 'average_reconstructed_c3_vs_c4_vs_cz_vs_fc3fc4'

file_type = 'png'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

file_type = 'eps'
filename = "{}.{}".format(name, file_type) 
plt.savefig(filename, format=file_type)

#%% END

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v23', type='model')
artifact_dir = artifact.download()