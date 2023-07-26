"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of the hierarchical vEEGNet (i.e. a EEGNet that work as a hierarchical VAE)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from . import vEEGNet, hierarchical_VAE

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

    def encode(self, x):
        z, mu, log_var, _ = self.h_vae.encoder.encode(x, return_distribution = True, return_shape = False)
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

    def dtw_comparison(self, x, distance_function = None):
        """
        Compute the DTW between x and the reconstructed version of x (obtained through the model)
        x : Tensor with eeg signal of shape B x 1 x C x T, with ( B = Batch dimension, 1 = Depth dimension, C = Number of channels, T = Time samples )
        """
        
        with torch.no_grad():
            output = self.forward(x)
            x_r = output[0]
            
            # Matrix to save all the DTW distance of shape B x C
            dtw_distance = np.zeros((x.shape[0], x.shape[2]))

            for i in range(x.shape[0]): # Cycle through batch dimension (i.e. eeg trial)
                eeg_trial = x[i]
                eeg_trial_r = x_r[i] 
                for j in range(x.shape[2]): # Cycle through channels
                    eeg_ch = eeg_trial[j]
                    eeg_ch_r = eeg_trial_r[j]

                    dtw_distance[i, j] = fastdtw(eeg_ch, eeg_ch_r, distance = distance_function)
            
            return distance_function


