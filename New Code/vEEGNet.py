"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of vEEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#%% Imports
import torch
from torch import nn

import EEGNet, MBEEGNet
import config_model

"""
%load_ext autoreload
%autoreload 2
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class vEEGnet(nn.Module):

    def __init__(self, config : dict):
        super().__init__()
        
        self.hidden_space = config['hidden_space']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Create Encoder

        # Convolutional section
        if config["type_encoder"] == 0:
            self.cnn_encoder = EEGNet(config['encoder_config']) 
        if config["type_encoder"] == 1:
            self.cnn_encoder = MBEEGNet(config['encoder_config']) 
        else:
            raise ValueError("type_encoder must be 0 (EEGNET) or 1 (MBEEGNet)")
        
        # Feed-Forward section
        n_input_neurons = self.compute_number_of_neurons(config['C'], config['T'])
        self.ff_encoder_mean = nn.Linear(n_input_neurons, self.hidden_space)
        self.ff_encoder_std = nn.Linear(n_input_neurons, self.hidden_space)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Create Decoder

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        # Other


    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.ff_encoder_mean(x)
        z_std = self.ff_encoder_std(x)

        return x

    def compute_number_of_neurons(self, C, T):
        """
        Compute the total number of neurons for the feedforward layer
        """

        x = torch.rand(1, 1, C, T)
        x = self.cnn_encoder(x)
        input_neurons = len(x.flatten())

        return input_neurons
