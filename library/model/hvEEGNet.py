"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of the hierarchical vEEGNet (i.e. a EEGNet that work as a hierarchical VAE)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

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

        label = self.forward(x)

        if return_as_index:
            predict_prob = torch.squeeze(torch.exp(label).detach())
            label = torch.argmax(predict_prob, dim = 1)

        return label

