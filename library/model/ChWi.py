"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Prototype of a (Ch)annel (Wi)se network
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
from torch import nn

from . import support_function as sf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Encoder

class ChWi_module_encoder(nn.Module) :
    def __init__(self, module_config : dict) :
        """
        Module for the Channel Wise network
        """
        super().__init__()
        
        # Check the config
        self.check_config(module_config)
        
        # Create the convolutional layer
        conv_layer = nn.Conv1d(in_channels = module_config['in_channels'], out_channels = module_config['out_channels'],
                               kernel_size = module_config['c_kernel'], padding = module_config['padding'], group = module_config['group']
                               )
        
        # Batch normalization
        batch_normalizer = nn.BatchNorm1d(module_config['out_channels']) if module_config['use_batch_normalization'] else nn.Identity()

        # Create activation
        activation = sf.get_activation(module_config['activation']) if module_config['activation'] is not None else nn.Identity()

        # Create pooling
        pooling = nn.AvgPool1d(module_config['p_kernel']) if module_config['p_kernel'] is not None else nn.Identity()

        self.chwi_module = nn.Sequential(
            conv_layer,
            batch_normalizer,
            activation,
            pooling,
        )

    def forward(self, x) :
        return self.chwi_module(x)

    def check_config(self, module_config : dict) :
        if 'print_var' not in module_config : module_config['print_var'] = False

        if 'in_channels' not in module_config : raise ValueError('in_channels must be in module_config')
        if 'out_channels' not in module_config : raise ValueError('out_channels must be in module_config')
        if 'c_kernel' not in module_config : raise ValueError('c_kernel (convolution kernel) must be in module_config')

        if 'group' not in module_config:
            module_config['group'] == 1

        if 'padding' not in module_config:
            module_config['padding'] == 0

        if 'use_batch_normalization' not in module_config:
            module_config['use_batch_normalization'] = False

        if 'activation' not in module_config:
            module_config['activation'] = None

        if 'p_kernel' not in module_config :
            module_config['p_kernel'] = None

class ChWi_encoder_v1(nn.Module) : 

    def __init__(self, config_list : list):
        """
        First version of the channel wise network. The model is composed of multiple chwi_module
        """
        super().__init__()
        
        # Variable to save the list of modules
        self.module_list = nn.Sequential()

        # Network creation (iterate through the modules config)
        for module_config in config_list:
            self.module_list.append(ChWi_module_encoder(module_config))

    def forward(self, x): 
        """
        x : EEG signal. The shape of x must be "B x 1 x T" with B = batch size, 1 = depth dimension, T = Time samples
        N.B. There must must be no EEG channel dimension
        """
        return self.module_list(x)
    
    def check_input(self, x):
        # TODO
        pass

    def reconstruct_multichannel_EEG(self, x, flatten : bool = False):
        """
        Compute, channel wise, the encoding through the ChWi modules.
        x : input x of shape "B x 1 x C x T" with B = batch size, 1 = depth dimension, C = Number of EEG channels, T = Time samples
        """
        
        # Compute the encode of a single channel to obtain the depth and temporal dimension
        tmp_encode = self.forward(x[:, :, 0, :])
        
        # Create the variable used to save the multi channels encoding
        x_encode = torch.zeros(x.size(0), tmp_encode.size(1), x.size(2), tmp_encode.size(2))

        for ch_idx in range(x.size(2)) :
            # Get data from EEG channel
            x_ch = x[:, :, ch_idx, :]
            
            # Encode the EEG channel
            x_encode[:, :, ch_idx, :] = self.forward(x_ch)

        if flatten :
            return x_encode.flatten(1)
        else :
            return x_encode

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Decoder

class ChWi_module_decoder(nn.Module) :
    def __init__(self, module_config : dict) :
        """
        Module for the Channel Wise network
        """
        super().__init__()

        # Create the convolutional layer
        conv_layer = nn.ConvTranspose1d(in_channels = module_config['in_channels'], out_channels = module_config['out_channels'],
                               kernel_size = module_config['c_kernel'], padding = module_config['padding'], group = module_config['group']
                               )
        
        # Batch normalization
        batch_normalizer = nn.BatchNorm1d(module_config['out_channels']) if module_config['use_batch_normalization'] else nn.Identity()

        # Create activation
        activation = sf.get_activation(module_config['activation']) if module_config['activation'] is not None else nn.Identity()

        # Create pooling
        upsample = nn.Upsample(scale_factor = module_config['scale_factor']) if module_config['scale_factor'] is not None else nn.Identity()

        self.chwi_module = nn.Sequential(
            upsample,
            conv_layer,
            batch_normalizer,
            activation,
        )

    def forward(self, x) :
        return self.chwi_module(x)

class ChWi_decoder_v1(nn.Module) : 

    def __init__(self, config_list : list):
        super().__init__()
        
        # Variable to save the list of modules
        self.module_list = nn.Sequential()

        # Network creation (iterate through the modules config)
        for module_config in config_list :
            self.module_list.append(ChWi_module_decoder(module_config))

    def forward(self, x): 
        """
        x : encode of a EEG signal. The shape of x must be "B x 1 x T" with B = batch size, 1 = depth dimension, T = length of temporal axis after encoding 
        N.B. There must must be no EEG channel dimension
        """
        return self.module_list(x)
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class ChWi_autoencoder(nn.Module) : 

    def __init__(self, config : dict) :
        super().__init__()
        
        # Create encoder and decoder
        self.encoder = ChWi_encoder_v1(config['encoder_config'])
        self.decoder = ChWi_decoder_v1(config['decoder_config'])

        # Hidden space map
        config['sample_layer_config']['input_depth'] = config['encoder_config'][-1]['out_channels']
        self.sample_layer = sf.sample_layer(-1, config['sample_layer_config'])

    def forward(self, x) :
        """
        Encode and reconstruct a single EEG channel.
        x : EEG signal. The shape of x must be "B x 1 x T" with B = batch size, 1 = depth dimension, T = Time samples
        N.B. There must must be no EEG channel dimension
        """
        
        # Encode the data
        x = self.encoder(x)
        
        # Sample from the hidden space
        z, mu, sigma = self.sample_layer(x)

        x_r = self.decoder(z)

        return x_r, z, mu, sigma

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class domain_predictor(nn.Module) : 

    def __init__(self, config : dict) :
        super().__init__()

        activation = sf.get_activation(config['activation']) if config['activation'] is not None else nn.Identity()
        input_layer  = nn.Linear(config['input_length'], config['input_length'] / 2)
        hidden_layer = nn.Linear(config['input_length'] / 2, config['input_length'] / 4)
        output_layer = nn.Linear(config['input_length'] / 4, config['n_domain'])

        self.predictor = nn.Sequential(
            input_layer,
            activation,
            hidden_layer,
            activation,
            output_layer,
            nn.Softmax() # TODO add dim
        )

    def forward(self, x) :
        return self.predictor(x)
