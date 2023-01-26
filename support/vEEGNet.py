#%% Imports
import torch
from torch import nn

import EEGNet
import VAE_EEGNet

"""
%load_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, 'support')
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Encoder

class encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        eegnet_config = EEGNet.get_EEGNet_config(config['C'], config['T'])
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_1']
        eegnet_config['stride'] = config['stride_1']
        self.eegnet_1 = nn.Sequential(
            EEGNet.EEGNet(config),
            nn.Flatten()
        ) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_2']
        eegnet_config['stride'] = config['stride_2']
        self.eegnet_2 = nn.Sequential(
            EEGNet.EEGNet(config),
            nn.Flatten()
        ) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_3']
        eegnet_config['stride'] = config['stride_3']
        self.eegnet_3 = nn.Sequential(
            EEGNet.EEGNet(config),
            nn.Flatten()
        ) 
        
               
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feedforward (i.e. mapping in the latent space)
        
        # Compute the number of input neurons of the feedforward layer
        x = torch.rand(1,1,config['C'], config['T'])
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)

        input_neurons = len(x1) + len(x2) + len(x3)
        
        self.fc_encoder = nn.Linear(input_neurons, config['hidden_space_dimension'])

    def forward(self, x):
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)

        x = torch.cat([x1, x2, x3], 1)

        x = self.fc_encoder(x)

        return x


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Decoder

    
