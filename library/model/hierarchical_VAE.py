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
        _, _, encoder_outputs_shape = self.encoder.encode(torch.rand(1, 1, config['encoder_config']['C'], config['encoder_config']['T']), return_shape = True)
        config['encoder_outputs_shape'] = encoder_outputs_shape

        # Create decoder
        self.decoder = hVAE_decoder(decoder_cell_list, config)

    def forward(self, x, h = None):
        # Encoder 
        z, mu, log_var, encoder_cell_outputs = self.encoder(x)
        
        # Decoder
        decoder_output = self.decoder(z, h, encoder_cell_outputs)
        x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = decoder_output

        # Add the mean and variance of the deepest layer to the list
        mu_list.insert(0, mu)
        log_var_list.insert(0, log_var)

        return x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list


class hvae_encoder(nn.Module):

    def __init__(self, encoder_cell_list : list, config : dict):
        super().__init__()

        self.encoder_modules = nn.ModuleList(encoder_cell_list)
        
        # Compute the shape of x after passing through the encoder
        tmp_x, _ = self.encode(torch.rand(1, 1, config['encoder_config']['C'], config['encoder_config']['T']))
        
        # Create the last layer to map the encoder output into the parameters of the normal distribution and sampled from it
        if 'hidden_space_dimension_list' in config: # Feedforward map (vector latent space)
            self.sample_layer_z = sf.sample_layer(tmp_x.shape, config, config['hidden_space_dimension_list'][0])
        else: # Convolutional map (matrix latent space)
            self.sample_layer_z = sf.sample_layer(tmp_x.shape, config)

        self.convert_logvar_to_var = config['convert_logvar_to_var']

    def forward(self, x):
        # Pass the data through the decoder
        x, cell_outputs = self.encode(x)
        z, mu, sigma = self.sample_layer_z(x)

        if self.convert_logvar_to_var: sigma = torch.exp(sigma)

        return z, mu, sigma, cell_outputs

    def encode(self, x, return_shape = False):
        # List with the output of each cell of the encoder
        cell_outputs = []
        output_shape = []
        
        for cell in self.encoder_modules: 
            x = cell(x)
            cell_outputs.append(x)
            if return_shape: output_shape.append(x.shape)
        
        if return_shape:
            return x, cell_outputs, output_shape
        else: 
            return x, cell_outputs


class hVAE_decoder(nn.Module):

    def __init__(self, decoder_cell_list : list, config : dict):
        super().__init__()
        
        tmp_decoder, tmp_sample_layers_z, tmp_sample_layers_z_given_x, tmp_features_combination_z, tmp_features_combination_decoder = self.build_decoder(decoder_cell_list, config)

        self.decoder = nn.ModuleList(tmp_decoder)
        self.sample_layers_z = nn.ModuleList(tmp_sample_layers_z)
        self.sample_layers_z_given_x = nn.ModuleList(tmp_sample_layers_z_given_x)
        self.features_combination_z = nn.ModuleList(tmp_features_combination_z)
        self.features_combination_decoder = nn.ModuleList(tmp_features_combination_decoder)

    def forward(self, z, h = None, encoder_cell_output = None):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # List to save the parameters of the distribution of the various layers
        mu_list = []
        log_var_list = []
        
        # Save the relative location (see section 3.2 paper NVAE)
        if encoder_cell_output is not None:
            delta_mu_list = []
            delta_log_var_list = []

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

        if h is not None:
            x = h
        else:
            x = torch.zeros_like(z)

        for i in range(len(self.decoder)):
            print(i, "DECODER")
            print(z.shape)
            # Combine the z with the output of the decoder
            # In the deepest layer (i == 0) combine h with z. If h is not passed a vector of zeros is used instead
            # TODO check after training the use of identity for the i == 0 if h is not passed
            x = self.features_combination_decoder[i](x, z)

            # Pass the data through the decoder cell
            x = self.decoder[i](x)
            
            # Sampling section
            # Note that in the last layer (i.e. the output I don't need to create a laten space and sampling from it... I'm only interested in the final results)
            if i != len(self.decoder) -1:
                # Sample from the distribution and save the results
                z, mu, log_var = self.sample_layers_z[i](x)
                mu_list.append(mu)
                log_var_list.append(log_var)
                
                # This section is used only during the training
                if encoder_cell_output is not None:
                    _, delta_mu, delta_log_var = self.sample_layers_z_given_x[i](self.features_combination_z[i](x, encoder_cell_output[-2 - i]))
                    # Note that the encoder_cell_output has ALL the output of the decoder, including the last (deepest) layer that is also the input of the decoder
                    # Due to the way the loop is structured I don't want the element corresponding to -1-i but the element corresponding to -2-i
                    # E.g if i = 0 I don't want - 1 - i = - 1 = last element of encoder_cell_output (Because encoder_cell_output[-1] is the input of the encoder but x is already pass through a decoder cell)
                    # For i = 0 what I want is element -2. So the correct indexing is -2-i

                    # "Correct" the normal distribution (section 3.2 NVAE paper)
                    z = self.sample_layers_z_given_x[i].reparametrize(mu + delta_mu, log_var + delta_log_var, return_as_matrix = True)
                
                    # Save the parameters
                    delta_mu_list.append(delta_mu)
                    delta_log_var_list.append(delta_log_var)
        
        if encoder_cell_output is None: # During generation
            return x, mu_list, log_var_list
        else: # During training
            return x, mu_list, log_var_list, delta_mu_list, delta_log_var_list
            
    def build_decoder(self, decoder_cell_list, config : dict):
        # Temporary list to save section of the decoder
        tmp_decoder = [] # Save the "main" module of the network
        tmp_sample_layers_z = [] # Sampling using during generation
        tmp_sample_layers_z_given_x = [] # Sampling using during training
        tmp_features_combination_z = [] # Save the modules used to combine the output of encoder and decoder that IT IS USED TO OBTAIN Z
        tmp_features_combination_decoder = [] # Save the modules used to combine the output of the decoder with the samples from the latens spaces

        # Temporary variables used to track shapes during network creations
        tmp_x = torch.rand(config['encoder_outputs_shape'][-1])
        tmp_z = torch.rand(config['encoder_outputs_shape'][-1])
        
        # List used to save the output of the decoder
        encoder_output_shape = config['encoder_outputs_shape']

        for i in range(len(decoder_cell_list)):
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
            # Features combination latent space (i.e. Decoder/encoder output combination, merge to create the z)
            # And Sample layer (layer that sample from the hidden space)

            if i > 0:
                # Nota that in the deepest layer the output of the encoder IS the input of the encoder
                # So there isn't necessity of combine anything

                # Create and save the module
                dec_enc_combination = sf.weighted_sum_tensor(encoder_output_shape[-1-i][1], tmp_x.shape[1], tmp_x.shape[1])
                tmp_features_combination_z.append(dec_enc_combination)
                
                # Notes on the -1-i index. The output of the encoder are saved in the order obtained from the encoder
                # So at position 0 we will have the output of the first cell of the encoder and at position -1 (the last) we will have the last output of the encoder
                # For the DECODER we need to use this information in reverse order, i.e. from last to first. 

                # Get the sample layer
                if 'hidden_space_dimension_list' in config: # Feedforward map (vector latent space)
                    sample_layer_z = sf.sample_layer(tmp_x.shape, config, config['hidden_space_dimension_list'][i])
                    sample_layer_z_given_x = sf.sample_layer(tmp_x.shape, config, config['hidden_space_dimension_list'][i])
                else: # Convolutional map (matrix latent space)
                    sample_layer_z = sf.sample_layer(tmp_x.shape, config)
                    sample_layer_z_given_x = sf.sample_layer(tmp_x.shape, config)

                tmp_sample_layers_z.append(sample_layer_z)
                tmp_sample_layers_z_given_x.append(sample_layer_z_given_x)

                # Pass the "data" through the modules
                tmp_encoder_output = torch.rand(encoder_output_shape[-1-i]) # Simulate the output of the ENCODER
                tmp_z, _, _ = sample_layer_z_given_x(dec_enc_combination(tmp_x, tmp_encoder_output))

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Features combination decoder
            # z/decoder combination (i.e. after sampling from latent space combine the result with the output of the decoder)
            
            # Create the module to combine features
            if i == 0:
                if config['use_h_in_decoder']: # See Fig. 2 in NVAE paper
                    z_dec_combination = sf.weighted_sum_tensor(tmp_x.shape[1], config['h_shape'][1], tmp_x.shape[1])
                    tmp_z = torch.rand(config['h_shape'])
                else:
                    z_dec_combination = sf.weighted_sum_tensor(tmp_x.shape[1], tmp_x.shape[1], tmp_x.shape[1])
                    tmp_z = torch.rand(tmp_x.shape)
            else:
                # Module creation
                z_dec_combination = sf.weighted_sum_tensor(tmp_x.shape[1], tmp_z.shape[1], tmp_x.shape[1])
            
            # Save the module
            tmp_features_combination_decoder.append(z_dec_combination)
            
            # Pass the "data" inside the module
            tmp_x = z_dec_combination(tmp_x, tmp_z)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Main network 

            # Get cell and obtain the output of the cell
            cell = decoder_cell_list[i]
            tmp_decoder.append(cell)
            
            # Pass the "data" through the cell
            tmp_x = cell(tmp_x)

        return tmp_decoder, tmp_sample_layers_z, tmp_sample_layers_z_given_x, tmp_features_combination_z, tmp_features_combination_decoder

