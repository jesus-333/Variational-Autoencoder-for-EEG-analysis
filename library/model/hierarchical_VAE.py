"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of the (deep) Hierchical VAE 
(https://arxiv.org/abs/2007.03898)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

from . import support_function

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class hVAE(nn.Module):

    def __init__(self, encoder_cell_list : list, decoder_cell_list : list):
        super().__init__()

        self.encoder = nn.ModuleList(*encoder_cell_list)
        self.decoder = nn.ModuleList(*decoder_cell_list)

    def forward(self, x):

        return x


class hvae_encoder(nn.Module):

    def __init__(self, encoder_cell_list : list, config : dict):
        super().__init__()

        self.encoder = nn.ModuleList(*encoder_cell_list)
        
        # Compute the number of depth channels after the input flow through all the encoder
        tmp_x, _ = self.encode(torch.rand(1, 1, config['C'], config['T']))
        last_depth = tmp_x.shape[1]
        
        # Map the output of the last layer in the mean and logsigma of the distribution
        self.map_to_distribution_parameters = map_to_distribution_parameters_with_convolution(last_depth, config['use_activation_last_layer'], config['activation'])

        self.convert_logvar_to_var = config['convert_logvar_to_var']

    def forward(self, x):
        # Pass the data through the decoder
        x, cell_output = self.encode(x)
        x = self.map_to_distribution_parameters(x)
        
        # Divide the output in mean (mu) and logvar (sigma). Note that the first half of the depth maps are used for the mean and the second half for the logvar
        mu, sigma = x.chunk(2, dim = 1)

        if self.convert_logvar_to_var: sigma = torch.exp(sigma)

        return mu, sigma, cell_output

    def encode(self, x):
        # List with the output of each cell of the encoder
        cell_output = []
        
        for cell in self.encoder: 
            x = cell(x)
            cell_output.append(x)

        return x, cell_output


class hVAE_decoder(nn.Module):

    def __init__(self, decoder_cell_list : list, config : dict):
        super().__init__()
        
        # Temporary list to save section of the decoder
        tmp_decoder = [] # Save the "main" module of the network
        tmp_map_parameters = []
        tmp_features_combination_z = [] # Save the modules used to combine the output of encoder and decoder
        tmp_features_combination_decoder = [] # Save the modules used to combine the output of the decoder with the samples from the latens spaces

        # Temporary input used to track shapes during network creations
        tmp_x = torch.rand(config['shape_decoder_input'])
        
        for cell in decoder_cell_list:
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Main network
            tmp_decoder.append(cell)
            tmp_x = cell(x)
            tmp_map_parameters.append(map_to_distribution_parameters_with_convolution(tmp_x.shape[1], config['use_activation_in_decoder_distribution_map'], config['activation']))
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

        self.decoder = nn.ModuleList(*tmp_decoder)
        self.map_to_distribution_parameters(*tmp_map_parameters)
        
        # Operation used to combine the various output of the encoder with the corresponding output of the decoder
        # Used only during training
        self.features_combination_z = nn.ModuleList(
            *[weighted_sum_features_map() for i in range(len(decoder_cell_list) - 1)]
        )
        
        # Combine the output of the cell of the decoder with the z of the various latent spaces
        self.features_combination_decoder = nn.ModuleList(
            *[weighted_sum_features_map() for i in range(len(decoder_cell_list))]
        )


    def forward(self, x, h = None, encoder_cell_output = None):
        if h is None: h = torch.zeros(x.shape).to(x.device)
        z = torch.cat([x, h], dim = 1)
        x = self.features_combination_decoder[0](z)

        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            
            # Map the output of the decoder cell in mean (mu) and logvar (sigma)
            mu, sigma = self.map_to_distribution_parameters[i](x).chunk(2, dim = 1)
            
            # This section is used only during the training
            if encoder_cell_output is not None:
                pass

        return x 
            

class weighted_sum_features_map(nn.Module):
    def __init__(self, depth_x1 : int, depth_x2 : int, depth_output):
        """
        Implement a 1 x 1 convolution that mix togheter two tensors. The convolution works along the depth dimension.
        Sinche the size of the kernel is (1 x 1) this operation is equal to a weighted sum between the features maps, with the weighted learned during the training
        """
        super().__init__()

        self.mix = nn.Conv2d(depth_x1 + depth_x2, depth_output, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.mix(x)
        return x

class map_to_distribution_parameters_with_convolution(nn.Module):
    def __init__(self, depth : int, use_activation : bool = False, activation : str = 'elu'):
        """
        Take in input a tensor of size B x D x H x W and return a tensor of size B x 2D x H x W
        It practically doubles the size of the depth. The first D map will be used as mean and the other will be used a logvar 
        """
        super().__init__()

        self.map = nn.Sequential(
            support_function.get_activation(activation) if use_activation else nn.Identity(),
            nn.Conv2d(depth, int(depth * 2), 1)
        )

    def forward(self, x):
        return self.map(x)
