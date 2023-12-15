"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of different classifier
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import torch
from torch import nn

from . import support_function
from . import hvEEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% 

class classifier_v1(nn.Module):

    def __init__(self, config : dict):
        """
        Simple classifier composed by feed forward layers
        """
        super().__init__()

        use_bias = config['use_bias']
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], use_droput_2d = False)

        input_layer = nn.Linear(config['input_size'], config['neurons_list'][0])

        hidden_layer = nn.ModuleList([input_layer])
        for i in range(len(config['neurons_list']) - 1):
            hidden_layer.append(dropout)
            hidden_layer.append(activation)
            hidden_layer.append(config['neurons_list'][i], config['neurons_list'][i + 1])

        self.clf = nn.Sequential(
            input_layer,
            *hidden_layer,
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        return self.clf(x)


class clf_model_v1(nn.Module):

    def __init__(self, config : dict):
        """
        Model composed by the encoder of hvEEGNet + classifier_v1
        """
        
        # Get the dictionary config
        config_hvEEGNet = config['config_hvEEGNet']
        config_clf = config['config_clf']
        
        # Create hvEEGNet and load weights
        tmp_hvEEGNet = hvEEGNet.hvEEGNet_shallow(config_hvEEGNet)
        tmp_hvEEGNet.load_state_dict(torch.load(config['path_weights'], map_location = 'cpu'))
        
        # Get the encoder and freeze the weights
        self.encoder = tmp_hvEEGNet.h_vae.encoder
        for param in self.encoder.parameters: param.require_grad = False

        # Classifier option and computation of input size
        self.use_only_mu_for_classification = config['use_only_mu_for_classification']
        tmp_x = torch.rand((1, 1, config_hvEEGNet['encoder_config']['C'], config_hvEEGNet['encoder_config']['T']))
        _, mu, log_var, _ = self.encoder(tmp_x)
        n_neurons = len(mu.flatten()) if self.use_only_mu_for_classification else len(mu.flatten()) + len(log_var.flatten())
        config_clf['input_size'] = n_neurons

        # Create the classifier
        self.clf = classifier_v1(config_clf)

    def forward(self, x):
        # Pass the data through the encoder
        _, mu, log_var, _ = self.encoder(x)
        
        if self.use_only_mu_for_classification:
            x = mu
        else:
            x = torch.cat([mu, log_var], dim = 1).flatten(1)

        label = self.clf(x)

        return label

