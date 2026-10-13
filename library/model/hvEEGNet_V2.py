"""
Implementation of the HVEEGNet_V2 model for EEG signal analysis.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

from . import EEGNet, hierarchical_VAE

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class hvEEGNet_V2(torch.torch.nn.Module):
    
    def __init__(self, config : dict):
        """
        """
        super().__init__()

        n_cells = config['n_cells']
        
        # Create lists of cells for the encoder and decoder
        encoder_cell_list = [EEGNet_cell(config['encoder_config']) for _ in range(n_cells)]
        decoder_cell_list = [EEGNet_cell(config['encoder_config']) for _ in range(n_cells)]

        self.h_vae = hierarchical_VAE.hVAE(encoder_cell_list, decoder_cell_list, config)

        if config['use_classifier'] :
            # TODO
            raise NotImplementedError("The classifier is not implemented yet. Please set 'use_classifier' to False in the configuration.")
        self.use_classifier = config['use_classifier']

    def forward(self, x : torch.Tensor):
        """

        """

        output = self.h_vae(x)
        x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = output

        if self.use_classifier:
            # TODO
            raise NotImplementedError("The classifier is not implemented yet. Please set 'use_classifier' to False in the configuration.")
        else:
            return x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list


class EEGNet_cell(torch.torch.nn.Module):
    
    def __init__(self, config : dict):
        """
        This class represent a single cell of the hvEEGNet_V2 model. Each cell is a single EEGNEt with residual connection.

        It takes as input a tensor of shape [B x 1 x C x T] and returns an output of the same shape. 
        Note that the output EEGNet is of shape [B x F x 1 x T'], where F and T' depends from the configuration of EEGNet.
        During the forware pass it is reshaped to [B x 1 x F x T'] to match the input shape of the next cell.

        But, since F and T' depends from the configuration of EEGNet, you have to ensure that the parameters of the EEGNet are set in a way that the output shape is compatible with the input shape of the next cell.
        For T' the easiest way to do this is to set `p_kernel_1` and `p_kernel_2` to -1, i.e. no pooling. Therefore T' = T.
        F instead is equals to the parameter `filter_2`. You can set it to the same value of C, i.e. the number of channels of the input, to ensure that the output shape is compatible with the input shape of the next cell.

        """
        super().__init__()
        
        # Create an EEGNet model based on the provided configuration
        self.eegnet = EEGNet.EEGNet(config)
        self.eegnet.flatten_output = False
        
        # (OPTIONAL) Debug print to check the output shape of the EEGNet
        if 'input_shape' in config :
            tmp_input = torch.randn(1, 1, config['input_shape'][0], config['input_shape'][1])
            tmp_output = self.forward(tmp_input)
            if tmp_output.shape != tmp_input.shape : print(f"Warning: the output shape of the EEGNet is {tmp_output.shape}, which is different from the input shape {tmp_input.shape}. Please check the configuration of the EEGNet to ensure that the output shape is compatible with the input shape of the next cell.") 

    def forward(self, x : torch.Tensor):
        """
        Forward pass of the EEGNet cell. It takse an input a tennsor of shape [B x 1 x C x T] and returns an output of the same shape.

        """

        # Pass the input through the EEGNet
        output = self.eegnet(x)
        
        # Reshape from [B x C x 1 x T] to [B x 1 x C x T]
        output = output.permute(0, 2, 1, 3)

        # Residual connection
        output = output + x

        return output
