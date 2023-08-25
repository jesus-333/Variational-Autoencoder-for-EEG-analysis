"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of the hierarchical vEEGNet (i.e. a EEGNet that work as a hierarchical VAE)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn
import numpy as np
from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw

from . import vEEGNet, hierarchical_VAE
from ..training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class hvEEGNet_shallow(nn.Module):
    def __init__(self, config : dict):
        super().__init__()

        encoder_cell_list, decoder_cell_list = self.build_cell_list(config)

        self.h_vae = hierarchical_VAE.hVAE(encoder_cell_list, decoder_cell_list, config)

        tmp_x = torch.rand((1, 1, config['encoder_config']['C'], config['encoder_config']['T']))
        _, mu_list, _, _, _ = self.h_vae(tmp_x)
        n_neurons = len(mu_list[0].flatten()) * 2
        
        if config['use_classifier']: 

            self.clf = nn.Sequential(
                nn.Linear(n_neurons, config['n_classes']),
                nn.LogSoftmax(dim = 1),
            )

        self.use_classifier = config['use_classifier']

    def forward(self, x):
        output = self.h_vae(x)
        x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = output

        if self.use_classifier: 
            z = torch.cat([mu_list[0], log_var_list[0]], dim = 1).flatten(1)
            label = self.clf(z)
            return x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, label
        else:
            return x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list

    def generate(self, z = None):
        return self.h_vae.generate(z)
    
    def reconstruct(self, x, no_grad = True):
        return self.h_vae.reconstruct(x, no_grad)

    def encode(self, x, return_distribution = True):
        z, mu, log_var, _ = self.h_vae.encoder.encode(x, return_distribution = return_distribution, return_shape = False)
        return z, mu, log_var

    def build_cell_list(self, config : dict):
        # List to save the cell of the encoder
        encoder_cell_list = []
        decoder_cell_list = []
        
        # Construct a standard vEEGNet
        tmp_vEEGNet = vEEGNet.vEEGNet(config)
        tmp_encoder = tmp_vEEGNet.cnn_encoder
        tmp_decoder = tmp_vEEGNet.decoder

        # Extract cells from ENCODER
        encoder_cell_list.append(tmp_encoder.temporal_filter)
        encoder_cell_list.append(tmp_encoder.spatial_filter)
        encoder_cell_list.append(tmp_encoder.separable_convolution)

        # Extract cells from DECODER 
        decoder_cell_list.append(tmp_decoder.separable_convolution_transpose)
        decoder_cell_list.append(tmp_decoder.spatial_convolution_transpose)
        decoder_cell_list.append(tmp_decoder.temporal_convolution_transpose)

        return encoder_cell_list, decoder_cell_list


    def classify(self, x, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """
        
        if self.use_classifier:

            output = self.forward(x)
            label = output[5]
    
            if return_as_index:
                predict_prob = torch.squeeze(torch.exp(label).detach())
                label = torch.argmax(predict_prob, dim = 1)
    
            return label
        
        else:
            raise ValueError("Model created without classifier")

    # def dtw_comparison(self, x, radius = 1, distance_function = None):
    #     """
    #     Compute the DTW between x and the reconstructed version of x (obtained through the model)
    #     x : Tensor with eeg signal of shape B x 1 x C x T, with ( B = Batch dimension, 1 = Depth dimension, C = Number of channels, T = Time samples )
    #     """
        
    #     with torch.no_grad():
    #         output = self.forward(x)
    #         x_r = output[0]
            
    #         # Matrix to save all the DTW distance of shape B x C
    #         dtw_distance = np.zeros((x.shape[0], x.shape[2]))

    #         for i in range(x.shape[0]): # Cycle through batch dimension (i.e. eeg trial)
    #             print(i)
    #             eeg_trial = x[i, 0]
    #             eeg_trial_r = x_r[i, 0]
    #             # The zero is needed to remove the depth dimension
    #             for j in range(x.shape[2]): # Cycle through channels
    #                 eeg_ch = eeg_trial[j].numpy()
    #                 eeg_ch_r = eeg_trial_r[j].numpy()
                    
    #                 distance, _ = fastdtw(eeg_ch, eeg_ch_r, radius = radius, dist = distance_function)
    #                 dtw_distance[i, j] = distance
            
    #         return dtw_distance
        
    def dtw_comparison_2(self, x, device = 'cpu'):
        """
        Compute the DTW between x and the reconstructed version of x (obtained through the model)
        x : Tensor with eeg signal of shape B x 1 x C x T, with ( B = Batch dimension, 1 = Depth dimension, C = Number of channels, T = Time samples )
        """
        
        with torch.no_grad():
            self.to(device)
            x = x.to(device)
            
            output = self.forward(x)
            x_r = output[0]
            
            # Matrix to save all the DTW distance of shape B x C
            dtw_distance = np.zeros((x.shape[0], x.shape[2]))
            
            use_cuda = True if device == 'cuda' else False
            recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

            for i in range(x.shape[2]): # Iterate through EEG Channels
                x_ch = x[:, :, i, :].swapaxes(1,2)
                x_r_ch = x_r[:, :, i, :].swapaxes(1,2)
                # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
                # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
                
                tmp_recon_loss = recon_loss_function(x_ch, x_r_ch)
                
                dtw_distance[:, i] = tmp_recon_loss.cpu()
                
            self.to('cpu')
            x = x.to('cpu')

            
            return dtw_distance



