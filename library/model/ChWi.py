"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Prototype of a (Ch)annel (Wi)se network
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
from torch import nn

from . import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class ChWi_module(nn.Module) :
    def __init__(self, module_config : dict) :
        """
        Module for the Channel Wise network
        """
        super().__init__()
        
        # Check the config
        self.check_config(module_config)
        
        # Create the convolutional layer
        conv_layer = nn.Conv1d(in_channels = module_config['in_channels'], out_channels = module_config['out_channels'],
                               kernel_size = module_config['c_kernel'], padding = module_config['padding'], group = module_config['group']
                               )
        
        # Batch normalization
        batch_normalizer = nn.BatchNorm1d(module_config['out_channels']) if module_config['use_batch_normalization'] else nn.Identity()

        # Create activation
        activation = support_function.get_activation(module_config['activation']) if module_config['activation'] is not None else nn.Identity()

        # Create pooling
        pooling = nn.AvgPool1d(module_config['p_kernel']) if module_config['p_kernel'] is not None else nn.Identity()

        self.chwi_module = nn.Sequential(
            conv_layer,
            batch_normalizer,
            activation,
            pooling,
        )

    def forward(self, x) :
        return self.chwi_module(x)

    def check_config(self, module_config : dict) :
        if 'print_var' not in module_config : module_config['print_var'] = False

        if 'in_channels' not in module_config : raise ValueError('in_channels must be in module_config')
        if 'out_channels' not in module_config : raise ValueError('out_channels must be in module_config')
        if 'c_kernel' not in module_config : raise ValueError('c_kernel (convolution kernel) must be in module_config')

        if 'group' not in module_config:
            module_config['group'] == 1

        if 'padding' not in module_config:
            module_config['padding'] == 0

        if 'use_batch_normalization' not in module_config:
            module_config['use_batch_normalization'] = False

        if 'activation' not in module_config:
            module_config['activation'] = None

        if 'p_kernel' not in module_config :
            module_config['p_kernel'] = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class ChWi_net_v1(nn.Module) : 

    def __init__(self, config : dict):
        """
        First version of the channel wise network. The model is composed of multiple chwi_module
        """
        super().__init__()
        
        # Variable to save the list of modules
        self.module_list = nn.Sequential()

        # Network creation (iterate through the modules config)
        for module_config in config['module_config_list'] :
            self.module_list.append(ChWi_module(module_config))

    def forward(self, x): 
        """
        x : EEG signal. The shape of x must be "B x 1 x T" with B = batch size, 1 = depth dimension, T = Time samples
        N.B. There must must be no EEG channel dimension
        """
        return self.module_list(x)
    
    def check_input(self, x):
        # TODO
        pass

    def reconstruct_multich_EEG(self, x):
        pass

