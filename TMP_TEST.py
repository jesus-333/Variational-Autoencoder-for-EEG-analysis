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
import scipy

def filter_signale(input_signal, order = 20):
    b, a = scipy.signal.butter(N = order, Wn = 15, btype = 'lowpass', fs = 128)
    output_signal = scipy.signal.filtfilt(b, a, input_signal)
    
    return output_signal

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

version = 135

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v{}'.format(version), type='model')
artifact_dir = artifact.download()

hidden_space_dimension = artifact.metadata['train_config']['hidden_space_dimension']

tmp_path = './artifacts/vEEGNet_trained-v{}'.format(version)
# network_weight_list = os.listdir(tmp_path)

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

#%% create Dataset (moabb) (ALL SUBJECT MERGE)

dataset_config = cf.get_moabb_dataset_config()
dataset_config['normalize_trials'] = True
dataset_config['subject_by_subject_normalization'] = False

train_data, validation_data = md.get_train_data(dataset_config)
test_data = md.get_test_data(dataset_config)

C = test_data[0][0].shape[1]
T = test_data[0][0].shape[2]
print("C = {}\nT = {}".format(C, T))

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
loader_list = [test_loader]

#%% create Dataset (moabb) (ALL SUBJECT DIVIDED)

dataset_list_train = []
dataset_list_test = []
dataloader_list_train = []
dataloader_list_test = []
subj_list = [1,2,3,4,5,6,7,8,9]

for i in range(len(subj_list)):
    subj = [subj_list[i]]
    print(subj)

    dataset_config = cf.get_moabb_dataset_config(subj)
    dataset_config['normalize_trials'] = True
    dataset_config['subject_by_subject_normalization'] = False
    
    tmp_train_data, validation_data = md.get_train_data(dataset_config)
    tmp_test_data = md.get_test_data(dataset_config)
    
    tmp_train_loader = torch.utils.data.DataLoader(tmp_train_data, batch_size = 32)
    tmp_test_loader = torch.utils.data.DataLoader(tmp_test_data, batch_size = 32)
    
    dataset_list_train.append(tmp_train_data)
    dataset_list_test.append(tmp_test_data)
    dataloader_list_train.append(tmp_train_loader)
    dataloader_list_test.append(tmp_test_loader)

#%%

import VAE_EEGNet

model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension,
                                use_reparametrization_for_classification = False,
                                print_var = True, tracking_input_dimension = True)

metrics_per_file = metrics.compute_metrics_given_path(model, loader_list,
                                                      tmp_path, device = 'cpu')

#%% Classification 1

accuracy_list = metrics_per_file['accuracy']
# for el in metrics_per_file:
#     tmp_metrics = np.asarray(el)
#     accuracy_list.append(tmp_metrics[:, 0])

accuracy = np.asarray(accuracy_list).T

avg_accuracy_per_file = np.mean(accuracy, 0)
max_avg_accuracy = np.max(avg_accuracy_per_file)

print(max_avg_accuracy)

#%% Classification 2

import VAE_EEGNet

hidden_space_dimension = 64
model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension,
                                use_reparametrization_for_classification = False,
                                print_var = True, tracking_input_dimension = True)

version = 21
epoch = 'END'
tmp_path = './artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch)
tmp_path = './TMP_File/model_BEST_TOTAL.pth'
model.load_state_dict(torch.load(tmp_path, map_location=torch.device('cpu')))

# Order of metrics
# accuracy, cohen_kappa, sensitivity, specificity, f1, confusion_matrix

metrics_train = metrics.compute_metrics(model, train_loader, 'cpu')
metrics_test = metrics.compute_metrics(model, test_loader, 'cpu')

print("Accuracy (TRAIN):\t{}".format(metrics_train[0]))
print("Accuracy (TEST) :\t{}".format(metrics_test[0]))

#%% Classification 3
# Run first the cell call  create Dataset (moabb) (ALL SUBJECT DIVIDED)

subj_list = [1,2,3,4,5,6,7,8,9]

accuracy_list_test = []
accuracy_list_train = []

C = 22
T = 513
hidden_space_dimension = 16
model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension,
                                use_reparametrization_for_classification = False,
                                print_var = True, tracking_input_dimension = True)

for i in range(len(subj_list)):
    subj = [subj_list[i]]
    print(subj)
    dataset_config = cf.get_moabb_dataset_config(subj)
    
    dataset_config['normalize_trials'] = True
    dataset_config['subject_by_subject_normalization'] = False

    train_data, validation_data = md.get_train_data(dataset_config)
    test_data = md.get_test_data(dataset_config)

    C = test_data[0][0].shape[1]
    T = test_data[0][0].shape[2]
    
    version = 21
    epoch = 'BEST_CLF'
    tmp_path = './artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch)
    model.load_state_dict(torch.load(tmp_path, map_location=torch.device('cpu')))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32)
    metrics_train = metrics.compute_metrics(model, train_loader, 'cuda')
    accuracy_list_train.append(metrics_train[0])
    
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    metrics_test = metrics.compute_metrics(model, test_loader, 'cuda')
    accuracy_list_test.append(metrics_test[0])


#%% PSD

import support_visualization as sv

version = 97
epoch_file = 'END'

# config = metrics.get_config_PSD()
config = dict(
    device = 'cuda',
    fs = 128, # Origina sampling frequency
    window_length = 2, # Length of the window in second
    second_overlap = 0
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

version = 124
epoch_file = 'END'

# config = metrics.get_config_PSD()
config = dict(
    device = 'cpu',
    fs = 128, # Origina sampling frequency
    window_length = 2, # Length of the window in second
    second_overlap = 1.75
)

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
state_dict = torch.load(path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

data_for_psd = test_data

psd_original_1, psd_reconstructed_1, f = metrics.psd_reconstructed_output(model, data_for_psd, 7, config)
psd_original_2, psd_reconstructed_2, f = metrics.psd_reconstructed_output(model, data_for_psd, 0, config)
psd_original_3, psd_reconstructed_3, f = metrics.psd_reconstructed_output(model, data_for_psd, 433, config)
psd_original_4, psd_reconstructed_4, f = metrics.psd_reconstructed_output(model, data_for_psd, 997, config)
print(len(psd_reconstructed_4))

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

version = 122
epoch_file = 'END'
path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
state_dict = torch.load(path)
state_dict = torch.load(path)


hidden_space_dimension = 64

model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension,
                                use_reparametrization_for_classification = False,
                                print_var = True, tracking_input_dimension = True)

channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
channel_list = np.asarray(channel_list)
idx_C3 = channel_list == 'C3'
idx_C4 = channel_list == 'C4'
idx_CZ = channel_list == 'Cz'

idx = 32
left_list = [0]
right_list = [1]
device = 'cpu'

model.load_state_dict(state_dict)

name = '2_FE_original vs reconstructed_C4_normalized_FILTERED'
name = 'TMP_File/PLOT TMP/' + name

sample = test_data[idx][0].unsqueeze(0)
label =  test_data[idx][1].unsqueeze(0)

if label == 0: label = 'LEFT'
if label == 1: label = 'RIGHT'
if label == 2: label = 'FOOT'
if label == 3: label = 'TONGUE'
print(label)

model.to(device)
x_r_mean, x_r_std, mu, log_var, _ = model(sample.to(device))

x = sample.cpu().squeeze().detach().numpy()
x_r = x_r_mean.cpu().squeeze().detach().numpy()
print(x.shape)
print(x_r.shape)

tmp_x = x[idx_C3].squeeze()
tmp_xr = x_r[idx_C3].squeeze()

tmp_x = (tmp_x - tmp_x.min())/(tmp_x.max() - tmp_x.min()) * 2 - 1
tmp_xr = (tmp_xr - tmp_xr.min())/(tmp_xr.max() - tmp_xr.min()) * 2 - 1

# FILTER SIGNAL
# tmp_x = filter_signale(tmp_x, order = 20)
# tmp_xr = filter_signale(tmp_xr, order = 20)

plt.figure(figsize = (15,10))
t = np.linspace(0, 4, T)

plt.plot(t, tmp_x, linestyle = 'solid' ,
          label = 'Original signal', linewidth = 2, color = 'grey', alpha = 0.55)
plt.plot(t, tmp_xr, linestyle = 'solid' ,
         label = 'Reconstructed Signal', linewidth = 2, color = 'black')


legend_properties = {'weight':'bold'}
plt.legend()

plt.xlabel("Time [s]", fontweight='bold')
plt.ylabel(r'Amplitude [$\mathbf{\mu V}$]', fontweight='bold')
plt.ylabel(r'Normalized Amplitude', fontweight='bold')
plt.xlim([0, 4])
# plt.ylim([0.029, 0.03])
plt.rcParams.update({'font.size': 18})
plt.rcParams["font.weight"] = "bold"
plt.tight_layout()
plt.grid(True)


# file_type = 'png'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)

# file_type = 'eps'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)
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

psd = psd_reconstructed_4
name = 'FE_30'

plt.figure(figsize = (15,10))

plt.plot(f, psd[idx_C3].squeeze(), linestyle = 'solid' ,
          label = 'C3', linewidth = 2, color = 'red')
plt.plot(f, psd[idx_C4].squeeze(), linestyle = 'dashed' ,
          label = 'C4', linewidth = 2, color = 'blue')
plt.plot(f, psd[idx_CZ].squeeze(), linestyle = 'dashdot' ,
          label = 'CZ', linewidth = 2, color = 'green')
plt.plot(f, ((psd[idx_foot_1] + psd[idx_foot_2]/2).squeeze()), linestyle = 'dotted' ,
          label = r'$\overline{FC34}$', linewidth = 2, color = 'black')

tmp_psd_1 = psd_reconstructed_3
tmp_psd_2 = psd_original_3
tmp_idx = idx_C4
name = 'FE_C4_PSD_Reconstructed vs original' 

# tmp_psd_1 = tmp_psd_1[tmp_idx].squeeze()
# tmp_psd_2 = tmp_psd_2[tmp_idx].squeeze()

# tmp_psd_1 = tmp_psd_1[idx_foot_1].squeeze() + tmp_psd_1[idx_foot_2].squeeze()
# tmp_psd_2 = tmp_psd_2[idx_foot_1].squeeze() + tmp_psd_2[idx_foot_2].squeeze()

# tmp_psd_1 = (tmp_psd_1 - tmp_psd_1.min()) / (tmp_psd_1.max() - tmp_psd_1.min())
# tmp_psd_2 = (tmp_psd_2 - tmp_psd_2.min()) / (tmp_psd_2.max() - tmp_psd_2.min())

# tmp_psd_1 /= tmp_psd_1.max()
# tmp_psd_2 /= tmp_psd_2.max()

# plt.plot(f, tmp_psd_1, linestyle = 'dashed' ,
#           label = 'Reconstructed', linewidth = 2, color = 'black')
# plt.plot(f, tmp_psd_2, linestyle = 'solid' ,
#           label = 'Original signal', linewidth = 2, color = 'grey')

legend_properties = {'weight':'bold'}
plt.legend()

plt.xlabel("Frequency [Hz]", fontweight='bold')
plt.ylabel(r'PSD [$\mathbf{\mu V^2}$/Hz]', fontweight='bold')
plt.rcParams.update({'font.size': 22})
# plt.ylim([0, 0.37])
plt.xlim([0, 40])
# plt.yscale('log')
plt.rcParams["font.weight"] = "bold"
plt.tight_layout()
plt.grid(True)

# file_type = 'png'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)

# file_type = 'eps'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)

#%%

version = 122
epoch_file = 'END'

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
state_dict = torch.load(path)

# hidden_space_dimension = int(state_dict['vae.decoder.decoder.fc_decoder.bias'].shape[0] / 4)

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

tmp_list = foot_list  # SIGNAL TO PLOT
name = '3_FE_average_reconstructed_c3_vs_c4_vs_cz_vs_fc3fc4'
name = 'TMP_File/PLOT TMP/' + name

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

tmp_max = np.max([avg_c3, avg_c4, avg_cz, avg_tongue])
tmp_min = np.min([avg_c3, avg_c4, avg_cz, avg_tongue])
avg_c3 = (avg_c3 - tmp_min)/(tmp_max - tmp_min) * 2 - 1
avg_c4 = (avg_c4 - tmp_min)/(tmp_max - tmp_min) * 2 - 1
avg_cz = (avg_cz - tmp_min)/(tmp_max - tmp_min) * 2 - 1
avg_tongue = (avg_tongue - tmp_min)/(tmp_max - tmp_min) * 2 - 1

plt.figure(figsize = (15,10))
t = np.linspace(0, 4, T)

plt.plot(t, avg_c3, linestyle = 'solid' ,
         label = 'C3', linewidth = 2, color = 'red', )
plt.plot(t, avg_c4, linestyle = 'dashed' ,
         label = 'C4', linewidth = 2, color = 'blue')
plt.plot(t, avg_cz, linestyle = 'dashdot' ,
         label = 'CZ', linewidth = 2, color = 'green')
plt.plot(t, avg_tongue, linestyle = 'dotted' ,
         label = r'$\overline{FC34}$', linewidth = 2, color = 'black')

plt.legend()
plt.xlabel("Time [s]", fontweight='bold')
plt.ylabel(r'Amplitude [$\mathbf{\mu V}$]', fontweight='bold')
plt.ylabel(r'Normalized Amplitude', fontweight='bold')
plt.xlim([0, 4])
plt.rcParams.update({'font.size': 22})
plt.tight_layout()
plt.grid(True)

# file_type = 'png'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)

# file_type = 'eps'
# filename = "{}.{}".format(name, file_type)
# plt.savefig(filename, format=file_type)

#%% Compute Reconstruction loss per subject

tmp_dataset_list = dataset_list_train
device = 'cpu'
version = 122
epoch_file = 'END'

path = 'artifacts/vEEGNet_trained-v{}/model_{}.pth'.format(version, epoch_file)
state_dict = torch.load(path)

hidden_space_dimension = int(state_dict['vae.decoder.decoder.fc_decoder.bias'].shape[0] / 4)

model = VAE_EEGNet.EEGFramework(C, T, hidden_space_dimension,
                                use_reparametrization_for_classification = False,
                                print_var = False, tracking_input_dimension = False)


model.load_state_dict(state_dict)
model.to(device)

loss_train = []
loss_test = []
std_train = []
std_test = []

recon_loss_function = torch.nn.MSELoss(reduction='none')
i = 0
for dataset_train, dataset_test in zip(dataset_list_train, dataset_list_test):
    print(i)
    i += 1
    # TRAIN
    x_r_mean, x_r_std, mu, log_var, _ = model(dataset_train[:][0].to(device))
    x   = dataset_train[:][0].cpu()
    x_r = x_r_mean.cpu()
    loss_per_trial = recon_loss_function(x, x_r).mean(dim = (1,2,3))
    loss_train.append(float(loss_per_trial.mean()))
    std_train.append(float(loss_per_trial.std()))
    torch.cuda.empty_cache()
    
    # TEST
    x_r_mean, x_r_std, mu, log_var, _ = model(dataset_test[:][0].to(device))
    x   = dataset_test[:][0].cpu()
    x_r = x_r_mean.cpu()
    loss_per_trial = recon_loss_function(x, x_r).mean(dim = (1,2,3))
    loss_test.append(float(loss_per_trial.mean()))
    std_test.append(float(loss_per_trial.std()))
    torch.cuda.empty_cache()
    

#%% END

run = wandb.init()
artifact = run.use_artifact('jesus_333/VAE_EEG/vEEGNet_trained:v23', type='model')
artifact_dir = artifact.download()
