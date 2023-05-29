"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of decoder based on EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class EEGNet_Decoder(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        use_bias = config['use_bias']
        D = config['D']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_activation(config['p_dropout'], config['use_dropout_2d'])

        
        # After the convolution layer reshape the array to this dimension
        self.dimension_reshape = config['dimension_reshape']
        
        self.separable_convolution_transpose = nn.Sequential(
            dropout,
            nn.Upsample(),
            activation,
            nn.BatchNorm2d(config['filter_2']),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
        )

        self.spatial_convolution_transpose = nn.Sequential(

        )

        self.temporal_convolution_transpose = nn.Sequential()
        
    def forward(self, z):
        return z

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Other function

