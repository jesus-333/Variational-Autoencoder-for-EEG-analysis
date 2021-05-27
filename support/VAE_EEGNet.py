# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import savemat, loadmat

from support_EEGNet import getParametersEncoder, getParametersDecoder
from DynamicNet import DynamicNet

#%% EEGNet VAE class

class EEGNetVAE(nn.Module):
    
    def __init__(self, C, T, hidden_space_dimension, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        self.encoder = EEGNetEncoder(C, T, hidden_space_dimension, print_var, tracking_input_dimension)
        output_shape_conv_encoder = self.encoder.conv_encoder.output_shape
        
        shape_original_input = (C, T)
        self.decoder = EEGNetDecoder(hidden_space_dimension, self.encoder.flatten_neurons, output_shape_conv_encoder, shape_original_input, print_var, tracking_input_dimension)
        
    def forward(self, x):
        z = self.encoder(x)
        
        x = self.decoder(z)
        
        return x
    
#%% EEGNet Encoder class
    
class EEGNetEncoder(nn.Module):
    
    def __init__(self, C, T, hidden_space_dimension, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        # Convolutional section
        parameters = getParametersEncoder(C = C, T = T)
        self.conv_encoder = DynamicNet(parameters, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        
        # Fully connect section
        self.flatten_neurons = self.conv_encoder.output_shape[1] * self.conv_encoder.output_shape[2] * self.conv_encoder.output_shape[3]
        self.fc_encoder = nn.Linear(self.flatten_neurons, hidden_space_dimension)
        
        
    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view([x.shape[0], -1])
        z = self.fc_encoder(x)
        
        return z
    
#%% EEGNet Decoder class

class EEGNetDecoder(nn.Module):
    
    def __init__(self, hidden_space_dimension, flatten_layer_dimension, conv_shape, shape_original_input, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        # Fully connect layer
        self.fc_decoder = nn.Linear(hidden_space_dimension, flatten_layer_dimension)
        
        # Convolutional layer
        parameters = getParametersDecoder(C = conv_shape[2], T = conv_shape[3])
        self.decoder = DynamicNet(parameters, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        self.shape_input_conv_decoder = list(conv_shape)
        self.shape_input_conv_decoder[0] = -1
        
        # Upsample layer
        self.upsample_layer = nn.Upsample(size = shape_original_input)
        
        
    def forward(self, z):
        x = self.fc_decoder(z)
        x = torch.reshape(x, self.shape_input_conv_decoder)
        
        x = self.decoder(x)
        
        x = self.upsample_layer(x)
        
        return x


