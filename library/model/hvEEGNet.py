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

        self.h_vae = hierarchical_VAE.hierarchical_VAE(encoder_cell_list, decoder_cell_list, config)

    def forward(self, x):
        x = self.h_vae(x)
        return x

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



