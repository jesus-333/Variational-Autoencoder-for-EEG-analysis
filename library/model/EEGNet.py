"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

from . import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Network declaration

class EEGNet(nn.Module):
    
    def __init__(self, config : dict):
        """
        Implementation of EEGNet in PyTorch.
        
        Note that compared to the original EEGNet the last layer (feedforward + classification) it is not present.
        The network offer the possibility to flatten the output (and obtain a vector of features of shape batch x n. features) or to mantain it as a tensor 

        The name of the variable respect the nomeclature of the original paper where it is possible.
        """

        super().__init__()
        
        use_bias = config['use_bias']
        D = config['D']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], config['use_dropout_2d'])
        
        # NOT USED
        self.C = config['C']
        self.T = config['T']
        
        # Used only if the EEGNet is used with time-frequency data
        # In this case the depth of the input is not 1
        depth_first_layer = config['depth_first_layer'] if 'depth_first_layer' in config else 1

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        
        # Block 1 - Temporal filters
        self.temporal_filter = nn.Sequential(
            nn.Conv2d(depth_first_layer, config['filter_1'], kernel_size = config['c_kernel_1'], 
                      padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_1']),
        )
        
        # Block 1 - Spatial filters
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_1'] * D),
            activation,
            nn.AvgPool2d(config['p_kernel_1']) if config['p_kernel_1'] is not None else nn.Identity(),
            dropout
        )

        # Block 2
        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            activation,
            nn.AvgPool2d(config['p_kernel_2']) if config['p_kernel_2'] is not None else nn.Identity(),
            dropout
        )
        
        if 'flatten_output' not in config: config['flatten_output'] = False
        self.flatten_output = config['flatten_output']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if 'print_var' not in config: config['print_var'] = False
        if config['print_var']:
            print("EEGNet Created. Number of parameters:")
            print("\tNumber of trainable parameters (Block 1 - Temporal) = {}".format(support_function.count_trainable_parameters(self.temporal_filter)))
            print("\tNumber of trainable parameters (Block 1 - Spatial)  = {}".format(support_function.count_trainable_parameters(self.spatial_filter)))
            print("\tNumber of trainable parameters (Block 2)            = {}\n".format(support_function.count_trainable_parameters(self.separable_convolution)))
            if 'input_size' in config: self.debug_shape(config['input_size'])

    def forward(self, x : torch.tensor) -> torch.tensor :
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 1 (temporal + spatial filters)
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 2 (separable convolution)
        x = self.separable_convolution(x)

        # (OPTIONAL) Flat the output
        if self.flatten_output: return x.flatten(1)
        else: return x

    def debug_shape(self, input_size):
        """
        Method that create a fake input and pass it through the network, showing after each pass the shape
        """

        print("Shape tracking across EEGNet")

        x = torch.randn(input_size)
        print("\tInput shape :\t\t\t", x.shape)

        x = self.temporal_filter(x)
        print("\tTemporal filter (conv):\t\t", x.shape)

        x = self.spatial_filter[0](x)
        print("\tSpatial fitler (conv) :\t\t", x.shape)

        x = self.spatial_filter[3](x)
        print("\tSpatial filter (pool) :\t\t", x.shape)

        x = self.separable_convolution[0](x)
        print("\tSeparable convolution (conv 1): ", x.shape)

        x = self.separable_convolution[1](x)
        print("\tSeparable convolution (conv 2): ", x.shape)

        x = self.separable_convolution[4](x)
        print("\tSeparable convolution (pool) :  ", x.shape)

        print("\t(OPTIONAL) Flatten :\t\t", x.flatten(1).shape, "\n")


class EEGNet_Classifier(nn.Module):

    def __init__(self, config : dict):
        super().__init__()
        """
        EEGNet + classifier
        """
        
        self.eegnet = EEGNet(config)
        
        input_neurons = self.compute_number_of_neurons(config['input_size'])
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config['n_classes']),
            nn.LogSoftmax(dim = 1)
        )

                
    def forward(self, x):
        x = self.eegnet(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, input_size):
        """
        Compute the total number of neurons for the feedforward layer
        """

        x = torch.rand(input_size)
        x = self.eegnet(x)
        input_neurons = len(x.flatten())

        return input_neurons
    
    def classify(self, x, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """

        label = self.forward(x)

        if return_as_index:
            predict_prob = torch.squeeze(torch.exp(label).detach())
            label = torch.argmax(predict_prob, dim = 1)

        return label

