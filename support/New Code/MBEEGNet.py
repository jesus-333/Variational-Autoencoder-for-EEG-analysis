"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Declaration of the decoder/classifier
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#%% Imports
import torch
from torch import nn

import EEGNet
import config_model

"""
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, 'support')
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Newtwork Declaration

class MBEEGNet(nn.Module):

    def __init__(self, config):
        """
        Implementation of MBEEGNet in PyTorch.
        
        The original paper is: 
        A Multibranch of Convolutional Neural Network Models for Electroencephalogram-Based Motor Imagery Classification
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8773854/
        """
        super().__init__()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        eegnet_config = config_model.get_EEGNet_config(config['C'])
        eegnet_config['flatten_output'] = True

        eegnet_config['c_kernel_1'] = config['temporal_kernel_1']
        eegnet_config['stride'] = config['stride_1']
        eegnet_config['dropout'] = config['dropout_1']
        self.eegnet_1 = EEGNet.EEGNet(eegnet_config) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_2']
        eegnet_config['stride'] = config['stride_2']
        eegnet_config['dropout'] = config['dropout_2']
        self.eegnet_2 = EEGNet.EEGNet(eegnet_config) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_3']
        eegnet_config['stride'] = config['stride_3']
        eegnet_config['dropout'] = config['dropout_3']
        self.eegnet_3 = EEGNet.EEGNet(eegnet_config) 

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feedforward (map to latent space/classification)


    def forward(self, x):
       return x

    def compute_number_of_neurons(self, C, T):
        """
        Compute the total number of neurons nee
        """

        x = torch.rand(1, 1, C, T)
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)
        input_neurons = len(x1.flatten()) + len(x2.flatten()) + len(x3.flatten())

        return input_neurons
