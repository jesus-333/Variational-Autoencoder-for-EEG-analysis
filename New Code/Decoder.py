"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of decoder based on EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn
import numpy as np

import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class EEGNet_Decoder(nn.Module):

    def __init__(self, config : dict):
        super().__init__()
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Parameter used during the creation of the networks
        use_bias = config['use_bias']
        D = config['D']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], config['use_dropout_2d'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Feed-Forward section of the decoder

        # After the convolution layer reshape the array to this dimension
        self.dimension_reshape = config['dimension_reshape']

        # Compute the number of neurons needed to obtain the shape defined by dimension_reshape
        n_ouput_neurons = np.abs(np.prod(self.dimension_reshape))
        
        # Defined feed-forward encoder
        self.ff_decoder = nn.Sequential(
            nn.Linear(config['hidden_space'], n_ouput_neurons),
            activation,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Convolutional section of the decoder

        self.separable_convolution_transpose = nn.Sequential(
            dropout,
            nn.Upsample(scale_factor = config['p_kernel_2']),
            activation,
            nn.BatchNorm2d(config['filter_2']),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, bias = use_bias),
        )

        self.spatial_convolution_transpose = nn.Sequential(
            dropout,
            nn.Upsample(scale_factor = config['p_kernel_1']),
            activation,
            nn.BatchNorm2d(config['filter_1'] * D),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'], kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
        )

        self.temporal_convolution_transpose = nn.Sequential(
            nn.BatchNorm2d(config['filter_1']),
            nn.Conv2d(config['filter_1'], 1, kernel_size = config['c_kernel_1'], bias = use_bias),
        )
        
    def forward(self, z):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Feed-Forward section
        
        # Feed-Forward decoder
        x = self.ff_decoder(z)

        # Reshape the output for the convolutional section
        x = torch.reshape(x, self.dimension_reshape)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Convolutional section
        x = self.separable_convolution_transpose(x)
        x = self.spatial_convolution_transpose(x)
        x = self.temporal_convolution_transpose(x)

        return x


