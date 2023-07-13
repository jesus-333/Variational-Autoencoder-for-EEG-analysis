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
        self.map_to_distribution_parameters = nn.Sequential(
            support_function.get_activation(config['activation']) if config['use_activation_last_layer'] else nn.Identity(),
            nn.Conv2d(last_depth, int(last_depth * 2), 1)
        )

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

    def __init__(self, decoder_cell_list, config : dict):
        super().__init__()

        self.decoder = nn.ModuleList(*decoder_cell_list)
        
        # Operation used to combine the various output of the encoder with the corresponding output of the decoder
        # Used only during training
        self.features_combination_z = nn.ModuleList(
            *[weighted_sum_features_map(config['depth_list_encoder'][i]) for i in range(len(decoder_cell_list) - 1)]
        )
        
        # Combine the output of the cell of the decoder with the z of the various latent spaces
        self.features_combination_decoder = nn.ModuleList(
            *[weighted_sum_features_map(config['depth_list_decoder'][i]) for i in range(len(decoder_cell_list))]
        )


    def forward(self, x, h = None, encoder_cell_output = None):
        if h is None: h = torch.zeros(x.shape).to(x.device)
        z = torch.cat([x, h], dim = 1)
        x = self.features_combination_decoder[0](z)
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
