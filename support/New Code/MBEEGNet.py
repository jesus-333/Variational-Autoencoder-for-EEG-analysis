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
        
        Respect the origina version this only extract a vector of features and NOT CLASSIFY them.
        To classify see MBEEGNet_Classifier.

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


    def forward(self, x):
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)

        x = torch.cat((x1, x2, x3), 1)

        return x


class MBEEGNet_Classifier(nn.Module):

    def __init__(self, config):
        """
        MBEEGNet + classifier
        """
        
        self.mbeegnet = MBEEGNet(config)
        
        input_neurons = self.compute_number_of_neurons(config['C'], config['T'])
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config['n_classes']),
            nn.LogSoftmax()
        )


    def forward(self, x):
        x = self.mbeegnet(x)

        x = self.classifier(x)

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
