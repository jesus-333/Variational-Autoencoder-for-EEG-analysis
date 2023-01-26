"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Network declaration

class EEGNet(nn.Module):
    
    def __init__(self, config):
        """
        Implementation of EEGNet in PyTorch.
        Read the GitHub page (https://github.com/jesus-333/EEGNet) to learn how to use this class
        """

        super().__init__()
        
        use_bias = config['use_bias']
        D = config['D']
        stride = config['stride'] if 'stride' in config else 1
        stride = 1
        padding = 0 if stride != 1 else 'same'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section

        self.temporal_filter = nn.Sequential(
            nn.Conv2d(1, config['filter_1'], kernel_size = config['c_kernel_1'], padding = padding, bias = use_bias, stride = stride),
            nn.BatchNorm2d(config['filter_1']),
        )
        
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_1'] * D),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_1']),
            nn.Dropout(config['dropout'])
        )

        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            config['activation'],
            nn.AvgPool2d(config['p_kernel_2']),
            nn.Dropout(config['dropout'])
        )


    def forward(self, x):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 1 (temporal + spatial filter)
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 2 (separable convolution)
        x = self.separable_convolution(x)
        

        return x


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# EEGNet config

def get_EEGNet_config(C = 22, T = 512):
    config = dict(
        # EEG Parameters
        C = C,
        T = T,
        D = 2,
        # Convolution: kernel size
        c_kernel_1 = (1, 64),
        c_kernel_2 = (C, 1),
        c_kernel_3 = (1, 16),
        # Convolution: number of filter
        filter_1 = 8,
        filter_2 = 16,
        #Pooling kernel
        p_kernel_1 = (1, 4),
        p_kernel_2 = (1, 8),
        # Other parameters
        activation = nn.ELU(),
        use_bias = False,
        dropout = 0.5
    )

    return config
