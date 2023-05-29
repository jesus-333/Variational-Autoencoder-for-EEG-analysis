"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of decoder based on EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


class EEGNet_Decoder(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

    def forward(self, z):
        return z
