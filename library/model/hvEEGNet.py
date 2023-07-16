"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of the hierarchical vEEGNet (i.e. a EEGNet that work as a hierarchical VAE)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

from . import EEGNet, Decoder_EEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class hvEEGNet_shallow(nn.Module):
    def __init__(self, config : dict):
        super().__init__()

    def forward(self, x):
        return x

    def build_encoder_cell(self, config : dict):
        # List to save the cell of the encoder
        encoder_cell_list = []

        # Create the EEGNet
        tmp_encoder = EEGNet.EEGNet(config['encoder_config']) 


