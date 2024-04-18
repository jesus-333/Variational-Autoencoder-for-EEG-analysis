"""
Contains support functions to download the dataset and create the models
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.fft as fft
import scipy.signal as signal

from ..config import config_model as cm
from ..dataset import preprocess as pp
from ..training import train_generic
from ..training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_dataset_and_model(dataset_config, model_name):
    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Create model
    if model_name == 'hvEEGNet_shallow':
        # hierarchical vEEGNet
        model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    elif model_name == 'vEEGNet':
        # classic vEEGNet
        model_config = cm.get_config_vEEGNet(C, T, hidden_space = -1, type_decoder = 0, type_encoder = 0)
        model_config['encoder_config']['p_kernel_1'] = None
        model_config['encoder_config']['p_kernel_2'] = (1, 10)
        model_config['use_classifier'] = False
        model_config['parameters_map_type'] = 0
        # Hidden space is set to -1 because with parameter map type = 0 do the parametrization trick through convolution (doubling the number of depth map)
    else:
        raise ValueError("Model name must be hvEEGNet_shallow or vEEGNet")

    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model_config['use_classifier'] = False
    model = train_generic.get_untrained_model(model_name, model_config)

    return train_dataset, validation_dataset, test_dataset, model


def compute_loss_dataset(dataset, model, device, batch_size = 32):
    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    recon_loss_matrix = np.zeros((len(dataset), len(dataset.ch_list)))
    
    with torch.no_grad():
        model.to(device)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        i = 0
        for x_batch, _ in dataloader:
            # Get the original signal and reconstruct it
            x = x_batch.to(device)
            x_r = model.reconstruct(x)
            
            # Compute the DTW channel by channels
            tmp_recon_loss = np.zeros((x_batch.shape[0], len(dataset.ch_list)))
            for j in range(x.shape[2]): # Iterate through EEG Channels
                x_ch = x[:, :, j, :].swapaxes(1, 2)
                x_r_ch = x_r[:, :, j, :].swapaxes(1, 2)
                # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
                # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
                
                tmp_recon_loss[:, j] = recon_loss_function(x_ch, x_r_ch).cpu()

            recon_loss_matrix[(i * batch_size):((i * batch_size) + x.shape[0]), :] = tmp_recon_loss
            i += 1

    return recon_loss_matrix


def compute_dtw_between_two_batch(batch_1, batch_2, use_cuda = True):
    """
    Given two batch of data compute the DTW between them.
    Batch must have shape B x 1 x C x T with B = Batch size, 1 = Depth size, C = EEG channels, T = Time samples
    """
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    # Compute the DTW channel by channels
    tmp_recon_loss = np.zeros((batch_1.shape[0], batch_1.shape[2]))
    for j in range(batch_1.shape[2]): # Iterate through EEG Channels
        x_1 = batch_1[:, :, j, :].swapaxes(1,2)
        x_2 = batch_2[:, :, j, :].swapaxes(1,2)
        # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
        # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
        
        tmp_recon_loss[:, j] = recon_loss_function(x_1, x_2).cpu()

    return tmp_recon_loss

def crop_signal(x, idx_ch, t_start, t_end, t_min, t_max):
    """
    Select a single channel according to idx_ch and crop it according to t_min and t_max provided in config
    t_start, t_end = Initial and final second of the x signal
    t_min, t_max = min and max to keep of the original signal
    """
    t      = np.linspace(t_start, t_end, x.shape[-1])
    x_crop = x.squeeze()[idx_ch, np.logical_and(t >= t_min, t <= t_max)]
    t_plot = t[np.logical_and(t >= t_min, t <= t_max)]

    return x_crop, t_plot


def compute_latent_space_different_resolution(model, x):
    """
    Reconstruct the EEG signal x using the contribution from different latent space
    model   = hvEEGNet to use
    x       = eeg signal of shape B x 1 x C x T
    """

    latent_space_to_ignore = [False, False, False]
    x_r_1 = model.h_vae.reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore).squeeze()

    latent_space_to_ignore = [True, False, False]
    x_r_2 = model.h_vae.reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore).squeeze()

    latent_space_to_ignore = [True, True, False]
    x_r_3 = model.h_vae.reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore).squeeze()
    
    return x_r_1.squeeze(), x_r_2.squeeze(), x_r_3.squeeze()

def compute_spectra_magnitude_and_phase(x, fs, keep_only_positive_frequency = True):
    spectra = fft.fftshift(fft.fft(x))
    magnitude = np.abs(spectra)
    phase = np.angle(spectra)
    f = fft.fftshift(fft.fftfreq(len(x), 1 / fs))
    
    if keep_only_positive_frequency:
        idx_f = f >= 0
        return magnitude[idx_f], phase[idx_f], f[idx_f]
    else:
        return magnitude, phase, f

def skip_training_run(subj, repetition):
    """
    List of training run to skip for each subject during the computation of the average reconstruction training loss in the analysis script 
    """
    if subj == 4 and (repetition == 17 or repetition == 18 or repetition == 19): return True
    if subj == 5 and repetition == 19: return True
    if subj == 8 and (repetition == 3 or repetition == 14 or repetition == 16): return True
    
    return False


def compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation = 2, skip_run = False, use_test_set = False):
    """
    As the name suggest compute the average and the std of the reconstruction error for each epoch/subject avereged across repetition.

    Important parameters:
        method_std_computation = 1: std along channels and average of std
        method_std_computation = 2: meand along channels and std of averages
        method_std_computation = 3: std of all the matrix (trials x channels)
        skip_run: if True avoid the use of some training run during the computation. The training run avoided presents some anomalies

    Output of the methods (note that each output is a dictionary with subject as key and for each subject there are dictionary with epoch as key):
        recon_loss_results_mean : Each element is a matrix of shape "n.trials x n.channels" averaged across the repetition
        recon_loss_results_std  : Each element is a vector of length n.trials" that represents the std for each trial in that specific epoch
        recon_loss_to_plot_mean : Each element is a float that represent the average error of the dataset for the specific epoch
        recon_loss_to_plot_std  : Each element is a float that represent the std of the error of the dataset for the specific epoch
    """
    recon_loss_results_mean = dict() # Save for each subject/repetition/epoch the average reconstruction error across channels
    recon_loss_results_std = dict() # Save for each subject/repetition/epoch the std of the reconstruction error across channels

    recon_loss_to_plot_mean = dict()
    recon_loss_to_plot_std = dict()

    if use_test_set: dataset_string = 'test'
    else: dataset_string = 'train'

    for subj in subj_list:
        recon_loss_results_mean[subj] = dict()
        recon_loss_results_std[subj] = dict()
        recon_loss_to_plot_mean[subj] = list()
        recon_loss_to_plot_std[subj] = list()
        
        for epoch in epoch_list:
            recon_loss_results_mean[subj][epoch] = 0
            recon_loss_results_std[subj][epoch] = 0
            
            valid_repetition = 0
            
            # Compute the mean and std of the error for each epoch across channels
            for repetition in repetition_list:
                if skip_training_run(subj, repetition) and skip_run:
                    print("Skip run {} subj {}".format(repetition, subj))
                    continue
                
                try:
                    path_load = 'Saved Results/repetition_hvEEGNet_{}/{}/subj {}/recon_error_{}_rep_{}.npy'.format(tot_epoch_training, dataset_string, subj, epoch, repetition)
                    tmp_recon_error = np.load(path_load)
                    
                    # recon_loss_results_mean[subj][epoch] += tmp_recon_error.mean(1)
                    recon_loss_results_mean[subj][epoch] += tmp_recon_error

                    if method_std_computation == 1:
                        recon_loss_results_std[subj][epoch] += tmp_recon_error.std(1)
                    elif method_std_computation == 2:
                        recon_loss_results_std[subj][epoch] += tmp_recon_error.mean(1)
                    elif method_std_computation == 3:
                        recon_loss_results_std[subj][epoch] += tmp_recon_error.std()
                    
                    valid_repetition += 1
                except:
                    print("File not found for subj {} - epoch {} - repetition {}".format(subj, epoch, repetition))

            recon_loss_results_mean[subj][epoch] /= valid_repetition
            recon_loss_results_std[subj][epoch] /= valid_repetition
            # Note that inside recon_loss_results_std[subj][epoch] there are vector of size n_trials
            
            recon_loss_to_plot_mean[subj].append(recon_loss_results_mean[subj][epoch].mean())
            if method_std_computation == 1:
                recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch].mean())
            elif method_std_computation == 2:
                recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch].std())
            elif method_std_computation == 3:
                recon_loss_to_plot_std[subj].append(recon_loss_results_std[subj][epoch])

    return recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std


def compute_average_spectra(data, nperseg, fs, idx_ch):
    # Create a variable to saved the average spectra for the various channels

    _, tmp_spectra = signal.welch(data.squeeze()[0][0][:], fs = fs, nperseg = nperseg)
    computed_spectra = np.zeros((len(data), len(tmp_spectra)))

    # Compute the average spectra
    for idx_trial in range(len(data)): # Cycle through eeg trials
        x = data[idx_trial]
        
        # Compute PSD
        f, x_psd = signal.welch(x.squeeze()[idx_ch, :].squeeze(), fs = fs, nperseg = nperseg)

        computed_spectra[idx_trial, :] = x_psd
    
    average_spectra = computed_spectra.mean(0)
    std_spectra = computed_spectra.std(0)

    return average_spectra, std_spectra, f
