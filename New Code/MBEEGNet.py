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

    def __init__(self, config : dict):
        """
        Implementation of MBEEGNet in PyTorch.
        
        With respect to the original version this only extract a vector of features and NOT CLASSIFY them.
        To classify see MBEEGNet_Classifier.

        The original paper is: 
        A Multibranch of Convolutional Neural Network Models for Electroencephalogram-Based Motor Imagery Classification
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8773854/
        """
        super().__init__()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        eegnet_config = config['eegnet_config']
        eegnet_config['flatten_output'] = True

        eegnet_config['c_kernel_1'] = config['temporal_kernel_1']
        eegnet_config['dropout'] = config['dropout_1']
        self.eegnet_1 = EEGNet.EEGNet(eegnet_config) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_2']
        eegnet_config['dropout'] = config['dropout_2']
        self.eegnet_2 = EEGNet.EEGNet(eegnet_config) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_3']
        eegnet_config['dropout'] = config['dropout_3']
        self.eegnet_3 = EEGNet.EEGNet(eegnet_config) 

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    def forward(self, x):
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)

        x = torch.cat((x1, x2, x3), 1)

        return x


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Classifier version

class MBEEGNet_Classifier(nn.Module):

    def __init__(self, config : dict):
        super().__init__()
        """
        MBEEGNet + classifier
        """
        
        self.mbeegnet = MBEEGNet(config)
        
        input_neurons = self.compute_number_of_neurons(config['C'], config['T'])
        self.classifier = nn.Sequential(
            nn.Linear(input_neurons, config['n_classes']),
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        x = self.mbeegnet(x)

        x = self.classifier(x)

        return x

    def compute_number_of_neurons(self, C, T):
        """
        Compute the total number of neurons for the feedforward layer
        """

        x = torch.rand(1, 1, C, T)
        x = self.mbeegnet(x)
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
