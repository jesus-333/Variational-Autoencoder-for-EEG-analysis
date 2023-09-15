# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.fft as fft

from ..config import config_dataset as cd
from ..config import config_model as cm
from ..dataset import preprocess as pp
from ..training import train_generic
from ..training.soft_dtw_cuda import SoftDTW
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_dataset_and_model(subj_list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Create model (hvEEGNet)
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model_config['use_classifier'] = False
    model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)

    return train_dataset, validation_dataset, test_dataset , model_hv


def compute_loss_dataset(dataset, model, device, batch_size = 32):
    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    recon_loss_matrix = np.zeros((len(dataset), len(dataset.ch_list)))
    
    with torch.no_grad():
        model.to(device)
        dataloader = DataLoader(dataset, batch_size = batch_size)
        i = 0
        for x_batch, _ in dataloader:
            # Get the original signal and reconstruct it
            x = x_batch.to(device)
            x_r = model.reconstruct(x)
            
            # Compute the DTW channel by channels
            tmp_recon_loss = np.zeros((x_batch.shape[0], len(dataset.ch_list)))
            for j in range(x.shape[2]): # Iterate through EEG Channels
                x_ch = x[:, :, j, :].swapaxes(1,2)
                x_r_ch = x_r[:, :, j, :].swapaxes(1,2)
                # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
                # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
                
                tmp_recon_loss[:, j] = recon_loss_function(x_ch, x_r_ch).cpu()

            recon_loss_matrix[(i * batch_size):((i * batch_size) + x.shape[0]), :] = tmp_recon_loss
            i += 1

    return recon_loss_matrix

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

def compute_spectra_magnitude_and_phase(x, fs):
    spectra = fft.fft(x)
    magnitude = np.abs(spectra)
    phase = np.angle(spectra)
    f = fft.fftfreq(len(x), 1 / fs)

    return magnitude, phase, f
