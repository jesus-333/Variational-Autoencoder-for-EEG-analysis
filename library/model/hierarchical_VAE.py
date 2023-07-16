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

from . import support_function as sf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class hVAE(nn.Module):

    def __init__(self, encoder_cell_list : list, decoder_cell_list : list, config : dict):
        super().__init__()
        # Create encoder
        self.encoder = hvae_encoder(encoder_cell_list, config)
        _, _, encoder_output_shape = self.encoder.encode(torch.rand(1, 1, config['C'], config['T']))
        config['encoder_output_shape'] = encoder_output_shape

        # Create decoder
        self.decoder = nn.ModuleList(*decoder_cell_list, config)

    def forward(self, x):

        return x


class hvae_encoder(nn.Module):

    def __init__(self, encoder_cell_list : list, config : dict):
        super().__init__()

        self.encoder_modules = nn.ModuleList(*encoder_cell_list)
        
        # Compute the number of depth channels after the input flow through all the encoder
        tmp_x, _ = self.encode(torch.rand(1, 1, config['C'], config['T']))
        last_depth = tmp_x.shape[1]
        
        # Map the output of the last layer in the mean and logsigma of the distribution
        self.map_to_distribution_parameters = sf.map_to_distribution_parameters_with_convolution(last_depth, config['use_activation_last_layer'], config['activation'])

        self.convert_logvar_to_var = config['convert_logvar_to_var']

    def forward(self, x):
        # Pass the data through the decoder
        x, cell_output = self.encode(x)
        x = self.map_to_distribution_parameters(x)
        
        # Divide the output in mean (mu) and logvar (sigma). Note that the first half of the depth maps are used for the mean and the second half for the logvar
        mu, sigma = x.chunk(2, dim = 1)

        if self.convert_logvar_to_var: sigma = torch.exp(sigma)

        return mu, sigma, cell_output

    def encode(self, x, return_shape = False):
        # List with the output of each cell of the encoder
        cell_output = []
        output_shape = []
        
        for cell in self.encoder_modules: 
            x = cell(x)
            cell_output.append(x)
            if return_shape: output_shape.append(x.shape)
        
        if return_shape:
            return x, cell_output, output_shape
        else: 
            return x, cell_output


class hVAE_decoder(nn.Module):

    def __init__(self, decoder_cell_list : list, config : dict):
        super().__init__()
        
        # TODO
        self.decoder = nn.ModuleList(*tmp_decoder)
        self.map_to_distribution_parameters(*tmp_map_parameters)
        
        # Operation used to combine the various output of the encoder with the corresponding output of the decoder
        # Used only during training
        self.features_combination_z = nn.ModuleList(
            *[sf.weighted_sum_features_tensor() for i in range(len(decoder_cell_list) - 1)]
        )
        
        # Combine the output of the cell of the decoder with the z of the various latent spaces
        self.features_combination_decoder = nn.ModuleList(
            *[sf.weighted_sum_features_tensor() for i in range(len(decoder_cell_list))]
        )


    def forward(self, x, h = None, encoder_cell_output = None):
        if h is not None: 
            z = torch.cat([x, h], dim = 1)
        else:
            z = x

        for i in range(len(self.decoder)):
            x = self.features_combination_decoder[i](z)
            x = self.decoder[i](x)
            
            # Map the output of the decoder cell in mean (mu) and logvar (sigma)
            mu, sigma = self.map_to_distribution_parameters[i](x).chunk(2, dim = 1)
            
            # This section is used only during the training
            if encoder_cell_output is not None:
                pass

        return x 
            
    def build_decoder(self, decoder_cell_list, config : dict):
        # Temporary list to save section of the decoder
        tmp_decoder = [] # Save the "main" module of the network
        tmp_map_parameters = []
        tmp_features_combination_z = [] # Save the modules used to combine the output of encoder and decoder that IT IS USED TO OBTAIN Z
        tmp_features_combination_decoder = [] # Save the modules used to combine the output of the decoder with the samples from the latens spaces

        # Temporary variables used to track shapes during network creations
        tmp_x = torch.rand(config['encoder_output_shape'][-1])
        tmp_z = torch.rand(config['encoder_output_shape'][-1])
        
        # List used to save the output of the decoder
        encoder_output_shape = config['encoder_output_shape']

        for i in range(len(decoder_cell_list)):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
            # Features combination latent space (i.e. Decoder/encoder output combination, merge to create the z)
            # And paramters map

            if i > 0:
                # Nota that in the deepest layer the output of the encoder IS the input of the encoder
                # So there isn't necessity of combine anything

                # Create and save the module
                dec_enc_combination = sf.weighted_sum_tensor(encoder_output_shape[-1-i][1], tmp_x.shape[1], tmp_x.shape[1])
                tmp_features_combination_z.append(dec_enc_combination)
                
                # Notes on the -1-i index. The output of the encoder are saved in the order obtained from the encoder
                # So at position 0 we will have the output of the first cell of the encoder and at position -1 (the last) we will have the last output of the encoder
                # For the DECODER we need to use this information in reverse order, i.e. from last to first. 

                # Get the parameters map
                if config['paramters_map_type'] == 0: # Convolution (i.e. matrix latent space)
                    paramters_map = sf.map_to_distribution_parameters_with_convolution(depth = tmp_x.shape[1],
                                                                                       use_activation = config['use_activation_in_decoder_distribution_map'], 
                                                                                       activation = config['activation'])
                elif config['paramters_map_type'] == 1: # Feedforward (i.e. vector latent space)
                    paramters_map = sf.map_to_distribution_parameters_with_vector(hidden_space_dimension = config['hidden_space_dimension_list'][i],
                                                                                  input_shape = tmp_x.shape,
                                                                                  use_activation = config['use_activation_in_decoder_distribution_map'], 
                                                                                  activation = config['activation'])
                else:
                    raise ValueError("config['paramters_map_type'] must have value 0 (convolution-matrix) or 1 (Feedforward-vector)")

                # Save the parameters map
                tmp_map_parameters.append(paramters_map)

                # Pass the "data" through the module
                tmp_encoder_output = torch.rand(encoder_output_shape[-1-i])
                tmp_z = paramters_map(tmp_x, tmp_encoder_output)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Features combination decoder
            # z/decoder combination (i.e. after sampling from latent space combine the result with the output of the decoder)
            
            # Create the module to combine features
            if i == 0:
                if config['use_h_in_decoder']: # See Fig. 2 in NVAE paper
                    # Module creation
                    z_dec_combination = sf.weighted_sum_tensor(tmp_x.shape[1], config['h_shape'][1], tmp_x.shape[1])
                    # Pass the "data"
                    tmp_x = z_dec_combination(tmp_x, torch.rand(config['h_shape']))
                else:
                    z_dec_combination = nn.Identity()
            else:
                # Module creation
                z_dec_combination = sf.weighted_sum_tensor(tmp_x.shape[1], tmp_z.shape[1], tmp_x.shape[1])
            
            # Save the module
            tmp_features_combination_decoder.append(z_dec_combination)
            
            # Pass the "data" inside the module
            tmp_x = z_dec_combination(tmp_x)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Main network 

            # Get cell and obtain the output of the cell
            cell = decoder_cell_list[i]
            tmp_decoder.append(cell)
            
            # Pass the "data" through the cell
            tmp_x = cell(tmp_x)

        return tmp_decoder, tmp_map_parameters, tmp_features_combination_z, tmp_features_combination_decoder

