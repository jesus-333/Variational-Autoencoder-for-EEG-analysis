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

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        eegnet_config = EEGNet.get_EEGNet_config(config['C'], config['T'])
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_1']
        eegnet_config['stride'] = config['stride_1']
        self.eegnet_1 = nn.Sequential(
            EEGNet.EEGNet(eegnet_config),
            nn.Flatten()
        ) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_2']
        eegnet_config['stride'] = config['stride_2']
        self.eegnet_2 = nn.Sequential(
            EEGNet.EEGNet(eegnet_config),
            nn.Flatten()
        ) 
        
        eegnet_config['c_kernel_1'] = config['temporal_kernel_3']
        eegnet_config['stride'] = config['stride_3']
        self.eegnet_3 = nn.Sequential(
            EEGNet.EEGNet(eegnet_config),
            nn.Flatten()
        ) 
        
               
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Feedforward (i.e. mapping in the latent space)
        
        # Compute the number of input neurons of the feedforward layer
        x = torch.rand(1,1,config['C'], config['T'])
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)
        input_neurons = len(x1.flatten()) + len(x2.flatten()) + len(x3.flatten())
        
        # Output layer (means and std of the hidden space)
        self.fc_encoder_mean = nn.Linear(input_neurons, config['hidden_space_dimension'])
        self.fc_encoder_std = nn.Linear(input_neurons, config['hidden_space_dimension'])

    def forward(self, x):
        x1 = self.eegnet_1(x)
        x2 = self.eegnet_2(x)
        x3 = self.eegnet_3(x)

        x = torch.cat([x1, x2, x3], 1)

        z_mu = self.fc_encoder_mean(x)
        z_log_var  = self.fc_encoder_std(x)

        return z_mu, z_log_var


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Decoder

class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        tmp_vae = VAE_EEGNet.EEGNetVAE(config['C'], config['T'], config['hidden_space_dimension'])

        self.decoder = tmp_vae.decoder

    def forward(self, x):
        
        x_r_mean, x_r_std = self.decoder(x)

        return x_r_mean, x_r_std


class decoder_simple(nn.Module):
    def __init__(self,):
        super().__init__()
        pass

    def forward(self,x):
        return x
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Decoder

class VAE(nn.Module):
    def __init__(self, config_encoder, config_decoder):
        super().__init__()
        
        self.encoder = Encoder(config_encoder)

        self.decoder = Decoder(config_decoder)


    def forward(self,x):
        z_mu, z_log_var = self.encoder(x)

        z = self.reparametrize(z_mu, z_log_var)
        
        x_mean, x_std = self.decoder(z)

        return x_mean, x_std, z_mu, z_log_var
    
    def reparametrize(self, mu, log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
        
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        
        return mu + sigma*z
