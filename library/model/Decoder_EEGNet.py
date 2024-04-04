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

from . import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class EEGNet_Decoder_Upsample(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Parameter used during the creation of the networks
        use_bias = config['use_bias']
        D = config['D']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], config['use_dropout_2d'])
        self.hidden_space = config['hidden_space']
        self.parameters_map_type = config['parameters_map_type']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Feed-Forward section of the decoder
        # Used only if parameters_map_type == 1 (i.e. if the map to the latent space is done through a feed-forward layer)

        self.dimension_reshape = config['dimension_reshape']

        if self.parameters_map_type == 1:
            # After the convolution layer reshape the array to this dimension

            # Compute the number of neurons needed to obtain the shape defined by dimension_reshape
            n_ouput_neurons = np.abs(np.prod(self.dimension_reshape))
            
            # Defined feed-forward encoder
            self.ff_decoder = nn.Sequential(
                nn.Linear(self.hidden_space, n_ouput_neurons),
                activation,
            )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Convolutional section of the decoder

        self.separable_convolution_transpose = nn.Sequential(
            dropout,
            nn.Upsample(scale_factor = config['p_kernel_2']) if config['p_kernel_2'] is not None else nn.Identity(),
            activation,
            nn.BatchNorm2d(config['filter_2']),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, bias = use_bias, padding = 'same'),
        )

        self.spatial_convolution_transpose = nn.Sequential(
            dropout,
            nn.Upsample(scale_factor = config['p_kernel_1']) if config['p_kernel_1'] is not None else nn.Identity(),
            activation,
            nn.BatchNorm2d(config['filter_1'] * D),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'], kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
        )

        self.temporal_convolution_transpose = nn.Sequential(
            nn.BatchNorm2d(config['filter_1']),
            nn.Conv2d(config['filter_1'], 1, kernel_size = config['c_kernel_1'], bias = use_bias, padding = 'same'),
        )

        if config['print_var']:
            print("Decoder-EEGNet Created. Number of parameters:")
            print("\tNumber of trainable parameters (Block 2 transpose)            = {}".format(support_function.count_trainable_parameters(self.separable_convolution_transpose)))
            print("\tNumber of trainable parameters (Block 1 - Spatial transpose)  = {}".format(support_function.count_trainable_parameters(self.spatial_convolution_transpose)))
            print("\tNumber of trainable parameters (Block 1 - Temporal transpose) = {}\n".format(support_function.count_trainable_parameters(self.temporal_convolution_transpose)))
            self.debug_shape()

    def forward(self, z : torch.tensor) -> torch.tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feed-Forward section
        
        if self.parameters_map_type == 1:
            # Feed-Forward decoder
            x = self.ff_decoder(z)

            # Reshape the output for the convolutional section
            x = torch.reshape(x, self.dimension_reshape)
        else:
            x = z
            
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        x = self.separable_convolution_transpose(x)
        x = self.spatial_convolution_transpose(x)
        x = self.temporal_convolution_transpose(x)

        return x

    def debug_shape(self):
        """
        Method that create a fake input and pass it through the network, showing the shape after each pass
        """

        print("Shape tracking across EEGNet based Decoder")
        
        if self.parameters_map_type == 1:
            x = torch.randn(1, self.hidden_space)
            print("\tInput shape :\t\t\t\t" , x.shape)

            x = self.ff_decoder(x)
            print("\tFF Decoder :\t\t\t\t" , x.shape)

            x = torch.reshape(x, self.dimension_reshape)
            print("\tAfter reshape :\t\t\t\t", x.shape)
        else:
            x = torch.randn(1, self.dimension_reshape[1], self.dimension_reshape[2], self.dimension_reshape[3])
            print("\tInput shape :\t\t\t\t" , x.shape)

        x = self.separable_convolution_transpose[1](x)
        print("\tSeparable convolution (upsample) :\t", x.shape)

        x = self.separable_convolution_transpose[4](x)
        print("\tSeparable convolution (conv 1) :\t", x.shape)

        x = self.separable_convolution_transpose[5](x)
        print("\tSeparable convolution (conv 2) :\t", x.shape)

        x = self.spatial_convolution_transpose[1](x)
        print("\tSpatial convolution (upsample) :\t", x.shape)

        x = self.spatial_convolution_transpose[4](x)
        print("\tSpatial convolution (conv) :\t\t", x.shape)

        x = self.temporal_convolution_transpose[1](x)
        print("\tTemporal convolution :\t\t\t", x.shape)


class EEGNet_Decoder_Transpose(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Parameter used during the creation of the networks
        use_bias = config['use_bias']
        D = config['D']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], config['use_dropout_2d'])
        self.hidden_space = config['hidden_space']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feed-Forward section of the decoder

        # After the convolution layer reshape the array to this dimension
        self.dimension_reshape = config['dimension_reshape']

        # Compute the number of neurons needed to obtain the shape defined by dimension_reshape
        n_ouput_neurons = np.abs(np.prod(self.dimension_reshape))
        
        # Defined feed-forward encoder
        self.ff_decoder = nn.Sequential(
            nn.Linear(self.hidden_space, n_ouput_neurons),
            activation,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section of the decoder

        self.separable_convolution_transpose = nn.Sequential(
            dropout,
            # nn.Upsample(scale_factor = config['p_kernel_2']),
            activation,
            nn.BatchNorm2d(config['filter_2']),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, bias = use_bias, stride = (1, 2)),
        )

        self.spatial_convolution_transpose = nn.Sequential(
            dropout,
            # nn.Upsample(scale_factor = config['p_kernel_1']),
            activation,
            nn.BatchNorm2d(config['filter_1'] * D),
            nn.ConvTranspose2d(config['filter_1'] * D, config['filter_1'], kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
        )

        self.temporal_convolution_transpose = nn.Sequential(
            nn.BatchNorm2d(config['filter_1']),
            nn.ConvTranspose2d(config['filter_1'], 1, kernel_size = config['c_kernel_1'], bias = use_bias, stride = (1, 2)),
        )

        if config['print_var']:
            print("Decoder-EEGNet Created. Number of parameters:")
            print("\tNumber of trainable parameters (Block 2 transpose)            = {}".format(support_function.count_trainable_parameters(self.separable_convolution_transpose)))
            print("\tNumber of trainable parameters (Block 1 - Spatial transpose)  = {}".format(support_function.count_trainable_parameters(self.spatial_convolution_transpose)))
            print("\tNumber of trainable parameters (Block 1 - Temporal transpose) = {}\n".format(support_function.count_trainable_parameters(self.temporal_convolution_transpose)))
            self.debug_shape()

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

    def debug_shape(self):
        """
        Method that create a fake input and pass it through the network, showing the shape after each pass
        """

        print("Shape tracking across EEGNet based Decoder")

        x = torch.randn(1, self.hidden_space)
        print("\tInput shape :\t\t\t\t" , x.shape)

        x = self.ff_decoder(x)
        print("\tFF Decoder :\t\t\t\t" , x.shape)

        x = torch.reshape(x, self.dimension_reshape)
        print("\tAfter reshape :\t\t\t\t", x.shape)

        # x = self.separable_convolution_transpose[1](x)
        # print("\tSeparable convolution (upsample) :\t", x.shape)

        x = self.separable_convolution_transpose[3](x)
        print("\tSeparable convolution (conv 1) :\t", x.shape)

        x = self.separable_convolution_transpose[4](x)
        print("\tSeparable convolution (conv 2) :\t", x.shape)

        # x = self.spatial_convolution_transpose[1](x)
        # print("\tSpatial convolution (pool) :\t\t", x.shape)

        x = self.spatial_convolution_transpose[3](x)
        print("\tSpatial convolution (conv) :\t\t", x.shape)

        x = self.temporal_convolution_transpose[1](x)
        print("\tTemporal convolution :\t\t\t", x.shape)
