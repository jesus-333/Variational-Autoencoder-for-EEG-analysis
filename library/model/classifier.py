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

        use_bias = config['use_bias'] if 'use_bias' in config else True
        activation = support_function.get_activation(config['activation'])
        dropout = support_function.get_dropout(config['prob_dropout'], use_droput_2d = False)

        input_layer = nn.Linear(config['input_size'], config['neurons_list'][0], bias = use_bias)

        hidden_layer = nn.ModuleList()
        for i in range(len(config['neurons_list']) - 1):
            hidden_layer.append(dropout)
            hidden_layer.append(activation)
            hidden_layer.append(nn.Linear(config['neurons_list'][i], config['neurons_list'][i + 1], bias = use_bias))

        self.clf = nn.Sequential(
            input_layer,
            *hidden_layer,
            nn.LogSoftmax(dim = 1)
        )

    def forward(self, x):
        return self.clf(x)


class classifier_model_v1(nn.Module):

    def __init__(self, config : dict):
        """
        Model composed by the encoder of hvEEGNet + classifier_v1
        """
        super().__init__()
        
        # Get the dictionary config
        config_hvEEGNet = config['config_hvEEGNet']
        config_clf = config['config_clf']
        
        # Create hvEEGNet and load weights
        tmp_hvEEGNet = hvEEGNet.hvEEGNet_shallow(config_hvEEGNet)
        tmp_hvEEGNet.load_state_dict(torch.load(config['path_weights'], map_location = 'cpu'))
        
        # Get the encoder and freeze the weights
        if config['freeze_encoder']:
            self.encoder = tmp_hvEEGNet.h_vae.encoder
            for param in self.encoder.parameters(): param.require_grad = False

        # Classifier option and computation of input size
        self.use_only_mu_for_classification = config_clf['use_only_mu_for_classification']
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
            x = mu.flatten(1)
        else:
            x = torch.cat([mu, log_var], dim = 1).flatten(1)

        y = self.clf(x) # Note that the clf network return the log probability of the n classes

        return y


    def classify(self, x, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """

        label = self.forward(x)

        if return_as_index:
            predict_prob = torch.squeeze(torch.exp(label).detach())
            label = torch.argmax(predict_prob, dim = 1)

        return label
